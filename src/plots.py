from typing import Union, List, Optional
import polars as pl
import altair as alt

def linechart(
        df: pl.DataFrame,
        x_axis: dict,
        y_axis: dict,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        palette: Optional[str] = "#69a481",
        time_unit: Optional[str] = "monthdate",
        scale: dict = {"type": "linear", "exponent": 0.4},
        title_color: Optional[str] = "#050315",
        add_labels: bool = False,
        text_format: Optional[str] = ".1f"
) -> alt.Chart:
    x_axis_name = x_axis.get("axis_name", x_axis["column_name"])
    y_axis_name = y_axis.get("axis_name", y_axis["column_name"])
    tick_format = y_axis.get("tick_format", ".1f")
    if title is not None:
        title_dict = {
            "text": title,
            "align": "left",
            "anchor": "start",
            "color": title_color
        }
        if subtitle is not None:
            title_dict["subtitle"] = subtitle
            title_dict["subtitleColor"] = title_color
    else:
        title_dict = {"text": ""}
    if "y_range" in y_axis.keys() and scale["type"]=="linear":
        scale_encoding = alt.Scale(
            type=scale["type"],
            exponent=scale["exponent"],
            domain=y_axis["y_range"]
        )
    else:
        scale_encoding = alt.Scale(
            type=scale["type"],
            exponent=scale["exponent"]
        )
    chart = alt.Chart(
        df,
        title=title_dict
    ).mark_line(
        color=palette
    ).encode(
        alt.X(
            x_axis["column_name"],
            title=x_axis_name,
            type="temporal",
            timeUnit=time_unit
        ),
        alt.Y(
            y_axis["column_name"],
            title=y_axis_name,
            scale=scale_encoding
        ).axis(format=tick_format),
        tooltip=[
            alt.Tooltip(
                x_axis["column_name"],
                title=x_axis["axis_name"]
            ),
            alt.Tooltip(
                y_axis["column_name"],
                title=y_axis["axis_name"],
                format=".2f"
            )
        ]
    )
    if add_labels:
        text = alt.Chart(
            df
        ).mark_text(
            align='center', dx=0, yOffset=-10, fontWeight="bold"
        ).encode(
            alt.X(
                x_axis["column_name"],
                title=x_axis_name,
                type="temporal",
                timeUnit=time_unit
            ).stack("zero"),
            alt.Y(
                y_axis["column_name"],
                title=y_axis_name,
                scale=scale_encoding
            ).axis(format=tick_format),
            text=alt.Text(
                y_axis["column_name"],
                format=text_format
            )
        ).properties(
            width=600
        )
        chart = chart + text
    return chart

def grouped_linechart(
        df: pl.DataFrame,
        x_axis: dict,
        y_axis: dict,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        palette: Optional[Union[str, List[str]]] = ["#69a481", "#7C1F31"],
        time_unit: Optional[str] = "monthdate",
        title_color: Optional[str] = "#050315",
        add_labels: bool = False,
        text_format: Optional[str] = ".1f"
) -> alt.Chart:
    x_axis_name = x_axis.get("axis_name", x_axis["column_name"])
    y_axis_name = y_axis.get("axis_name", y_axis["column_name"])
    tick_format = y_axis.get("tick_format", ".1f")
    if title is not None:
        title_dict = {
            "text": title,
            "align": "left",
            "anchor": "start",
            "color": title_color
        }
        if subtitle is not None:
            title_dict["subtitle"] = subtitle
            title_dict["subtitleColor"] = title_color
    else:
        title_dict = {"text": ""}
    chart = alt.Chart(
        df,
        title=title_dict
    ).mark_line().encode(
        alt.X(
            x_axis["column_name"],
            title=x_axis_name,
            type="temporal",
            timeUnit=time_unit
        ),
        alt.Y(
            y_axis["column_name"],
            title=y_axis_name
        ).axis(format=tick_format),
        color=alt.Color(
            y_axis["group_by"],
            scale=alt.Scale(range=palette),
            title=y_axis["group_by"].title()
        ),
        tooltip=[
            alt.Tooltip(
                y_axis["group_by"],
                title=y_axis["group_by"].title()
            ),
            alt.Tooltip(
                x_axis["column_name"],
                title=x_axis["axis_name"]
            ),
            alt.Tooltip(
                y_axis["column_name"],
                title=y_axis["axis_name"],
                format=".2f"
            )
        ]
    )
    if add_labels:
        text = alt.Chart(
            df
        ).mark_text(
            align='center', dx=0, yOffset=-10, fontWeight="bold"
        ).encode(
            alt.X(
                x_axis["column_name"],
                title=x_axis_name,
                type="temporal",
                timeUnit=time_unit
            ).stack("zero"),
            alt.Y(
                y_axis["column_name"],
                title=y_axis_name
            ).axis(format=tick_format),
            text=alt.Text(
                y_axis["column_name"],
                format=text_format
            )
        ).properties(
            width=600
        )
        chart = chart + text
    return chart