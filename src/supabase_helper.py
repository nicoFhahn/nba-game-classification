import polars as pl
from datetime import datetime
from dateutil.relativedelta import relativedelta    


def fetch_distinct_column(supabase, table_name, column_name, page_size=10000):
    all_values = set()
    start = 0
    while True:
        response = (
            supabase
            .table(table_name)
            .select(column_name)
            .order("id")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        # Extract the column values and add to set (automatically deduplicates)
        all_values.update(row[column_name] for row in batch if row.get(column_name) is not None)
        start += page_size
    return list(all_values)


def fetch_entire_table(supabase, table_name, page_size=10000, sort=False):
    all_rows = []
    start = 0
    while True:
        if not sort:
            response = (
                supabase
                .table(table_name)
                .select("*")
                .range(start, start + page_size - 1)
                .execute()
            )
        else:
            response = (
                supabase
                .table(table_name)
                .select("*")
                .order("id")  # Order by primary key or another unique column
                .range(start, start + page_size - 1)
                .execute()
            )
        batch = response.data
        if not batch:
            break
        all_rows.extend(batch)
        start += page_size
    return pl.DataFrame(all_rows)

def fetch_month_data(supabase, table_name, date_obj, date_column='date', page_size=10000):
    """
    Fetch all rows for a given month from a Supabase table.
    
    Args:
        supabase: Supabase client instance
        table_name: Name of the table to query
        date_obj: A date object (e.g., date.today(), date(2026, 1, 15))
        date_column: Name of the date/timestamp column to filter on (default: 'created_at')
        page_size: Number of rows to fetch per request (default: 10000)
    
    Returns:
        polars.DataFrame containing all rows for the specified month
    """
    # Get the first day of the month
    start_date = datetime(date_obj.year, date_obj.month, 1)
    # Get the first day of the next month
    end_date = start_date + relativedelta(months=1)
    
    # Format dates for Supabase query (ISO format)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    
    all_rows = []
    start = 0
    
    while True:
        response = (
            supabase
            .table(table_name)
            .select("*")
            .gte(date_column, start_str)
            .lt(date_column, end_str)
            .range(start, start + page_size - 1)
            .execute()
        )
        
        batch = response.data
        if not batch:
            break
        
        all_rows.extend(batch)
        start += page_size
    
    return pl.DataFrame(all_rows)


def fetch_filtered_table(supabase, table_1, table_2, col_1, col_2, col_1_value=None, chunk_size=50):
    # Step 1: Fetch all rows from table_1 where col_1 == max(col_1) or col_1 == col_1_value
    if col_1_value is None:
        max_response = (
            supabase
            .table(table_1)
            .select(col_1)
            .order(col_1, desc=True)
            .limit(1)
            .execute()
        )
        if not max_response.data:
            return pl.DataFrame(), pl.DataFrame()
        filter_value = max_response.data[0][col_1]
    else:
        filter_value = col_1_value

    # Fetch all rows from table_1
    table_1_rows = []
    start = 0
    page_size = 1000
    while True:
        response = (
            supabase
            .table(table_1)
            .select("*")
            .eq(col_1, filter_value)
            .order(col_2)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        table_1_rows.extend(batch)
        start += page_size

    table_1_df = pl.DataFrame(table_1_rows)
    if table_1_df.is_empty():
        return table_1_df, pl.DataFrame()

    col_2_values = table_1_df[col_2].unique().to_list()

    # Step 2: Fetch table_2 in chunks to avoid query timeouts
    table_2_rows = []

    for i in range(0, len(col_2_values), chunk_size):
        chunk = col_2_values[i:i + chunk_size]
        start = 0
        while True:
            response = (
                supabase
                .table(table_2)
                .select("*")
                .in_(col_2, chunk)
                .order(col_2)
                .range(start, start + page_size - 1)
                .execute()
            )
            batch = response.data
            if not batch:
                break
            table_2_rows.extend(batch)
            start += page_size

    return table_1_df, pl.DataFrame(table_2_rows).sort(col_2)
