import json

# 定义一个函数，用于将输入的 JSON 数据转换为指定的格式

def convert_lmdb_to_target_format(input_data):
    output_list = []
    for block in input_data["blocks_info"]:
        converted = {
            "index": block["index"],
            "filename": block["source"],
            # "height": block["original_size"][1],  # 原图高度
            # "width": block["original_size"][0],  # 原图宽度
            # "i": block["position"][1],  # 行索引（纵向位置）
            # "j": block["position"][0]  # 列索引（横向位置）
            "height": block["original_size"][0],  # 原图高度 ← 交换索引
            "width": block["original_size"][1],  # 原图宽度 ← 交换索引
            "i": block["position"][0],  # 行索引 ← 交换顺序
            "j": block["position"][1]  # 列索引 ← 交换顺序

        }
        output_list.append(converted)
    return output_list


# 示例使用
if __name__ == "__main__":

    with open(r"E:\dataset_mdb\fig2_lmdb\dataset_metadata.json") as f:
        input_data = json.load(f)
    # 执行转换
    converted_data = convert_lmdb_to_target_format(input_data)

    # 输出结果（带缩进的JSON格式）
    # print(json.dumps(converted_data, indent=2))
    with open(r"E:\dataset_mdb\fig2_lmdb\dataset_metadata_change.json", "w") as f:
        json.dump(converted_data, f)