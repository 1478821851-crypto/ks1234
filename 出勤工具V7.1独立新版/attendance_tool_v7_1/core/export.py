import io
import pandas as pd

def make_excel(df):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="出勤结果"
        )

        worksheet = writer.book[
            "出勤结果"
        ]

        worksheet.column_dimensions["A"].width = 22
        worksheet.column_dimensions["B"].width = 12
        worksheet.column_dimensions["C"].width = 12
        worksheet.column_dimensions["D"].width = 12
        worksheet.column_dimensions["E"].width = 12

    output.seek(0)

    return output
