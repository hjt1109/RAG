from excel_processor import ExcelProcessor
import pandas as pd
excel_processor = ExcelProcessor()
file_path = "/home/hjt/mountpoint/rag_project/组件信息表.xlsx"
df = pd.read_excel(file_path)


class TestProcessDataComponent:
    def test_process_data_component(self):
        data_component = excel_processor._process_data_component(df)
        return  data_component

if __name__ == '__main__':
    test_process_data_component = TestProcessDataComponent()
    test_process_data_component.test_process_data_component()
