import polars as pl
from datetime import datetime
from dateutil.relativedelta import relativedelta


def fetch_distinct_column(supabase, function_name, page_size=10000):
    all_ids = []
    start = 0

    while True:
        r = (
            supabase.rpc(function_name)
            .range(start, start + page_size - 1)
            .execute()
        )

        batch = [row['id_value'] for row in r.data]
        if not batch:
            break

        all_ids.extend(batch)
        start += page_size
    return list(all_ids)


def fetch_distinct_column_filtered(supabase, function_name, season_id, page_size=1000):
    """
    Fetch distinct values via RPC, filtered by season_id.
    """
    all_ids = []
    start = 0

    while True:
        r = (
            supabase.rpc(
                function_name, 
                {"target_season_id": season_id}
            )
            .range(start, start + page_size - 1)
            .execute()
        )

        batch = [row['id_value'] for row in r.data]
        if not batch:
            break

        all_ids.extend(batch)
        start += page_size
    return list(all_ids)


def fetch_distinct_column_in(supabase, table_name, column_name, filter_column, filter_values, page_size=10000,
                             chunk_size=50):
    """
    Fetch distinct values of `column_name` from `table_name`, but only for rows
    where `filter_column` is in `filter_values`. Queries in chunks to stay within
    Supabase URL-length limits.
    """
    all_values = set()
    for i in range(0, len(filter_values), chunk_size):
        chunk = filter_values[i:i + chunk_size]
        start = 0
        while True:
            response = (
                supabase
                .table(table_name)
                .select(column_name)
                .in_(filter_column, chunk)
                .range(start, start + page_size - 1)
                .execute()
            )
            batch = response.data
            if not batch:
                break
            all_values.update(row[column_name] for row in batch if row.get(column_name) is not None)
            start += page_size
    return list(all_values)


def fetch_entire_table(supabase, table_name, page_size=10000, sort=False, columns="*", filter_func=None):
    """
    Fetch rows from a Supabase table with optional column selection and filtering.
    
    Args:
        supabase: Supabase client instance
        table_name: Table to query
        page_size: Rows per paginated request
        sort: Whether to sort by "id"
        columns: Comma-separated string of columns or list of column names (default: "*")
        filter_func: A callable that takes a Supabase query object and returns it with filters applied.
    """
    if isinstance(columns, list):
        columns = ",".join(columns)
        
    all_rows = []
    start = 0
    while True:
        query = supabase.table(table_name).select(columns)
        
        if filter_func:
            query = filter_func(query)
            
        if sort:
            query = query.order("id")
            
        response = query.range(start, start + page_size - 1).execute()
        
        batch = response.data
        if not batch:
            break
        all_rows.extend(batch)
        start += page_size
    return pl.DataFrame(all_rows)


def fetch_month_data(supabase, table_name, date_obj, date_column='date', page_size=10000, columns="*", filter_func=None):
    """
    Fetch all rows for a given month from a Supabase table with optional column selection and filtering.
    
    Args:
        filter_func: A callable that takes a Supabase query object and returns it with filters applied.
    """
    if isinstance(columns, list):
        columns = ",".join(columns)

    start_date = datetime(date_obj.year, date_obj.month, 1)
    end_date = start_date + relativedelta(months=1)

    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    all_rows = []
    start = 0

    while True:
        query = (
            supabase
            .table(table_name)
            .select(columns)
            .gte(date_column, start_str)
            .lt(date_column, end_str)
        )
        
        if filter_func:
            query = filter_func(query)
            
        response = query.range(start, start + page_size - 1).execute()

        batch = response.data
        if not batch:
            break

        all_rows.extend(batch)
        start += page_size

    return pl.DataFrame(all_rows)


def fetch_filtered_table(supabase, table_1, table_2, col_1, col_2, col_1_value=None, chunk_size=50, columns_1="*", columns_2="*"):
    """
    Fetch filtered data from two tables with optional column selection.
    
    Args:
        columns_1: Columns for table_1 (list or comma-separated string)
        columns_2: Columns for table_2 (list or comma-separated string)
    """
    if isinstance(columns_1, list):
        columns_1 = ",".join(columns_1)
    if isinstance(columns_2, list):
        columns_2 = ",".join(columns_2)

    # Ensure col_1 is in columns_1 if not "*"
    if columns_1 != "*" and col_1 not in columns_1.split(","):
        columns_1 = f"{col_1},{columns_1}" if columns_1 else col_1

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
            .select(columns_1)
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

    # Step 2: Fetch table_2 in chunks
    table_2_rows = []

    for i in range(0, len(col_2_values), chunk_size):
        chunk = col_2_values[i:i + chunk_size]
        start = 0
        while True:
            response = (
                supabase
                .table(table_2)
                .select(columns_2)
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
