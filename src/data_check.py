from src.utils import load_parquet_data
from tqdm import tqdm
import json  # 1. 导入 json 模块


data_path = "dataset/NaturalQuestionsShort.parquet"
datas = load_parquet_data(data_path)
res_sum = []
for i, data in enumerate(tqdm(datas)):
    question = data['question']
    context = data['context']
    gold_ans = data['answers']
    # 🔽 核心修改：在这里将可能是 ndarray 的字段转换为 list
    if hasattr(context, 'tolist'):
        context = context.tolist()
    if hasattr(gold_ans, 'tolist'):
        gold_ans = gold_ans.tolist()
        
    res = {'context': context, 'question': question, 'gold_ans': gold_ans}
    res_sum.append(res)

# 写入 JSON 文件 (纯英文数据可以不加 ensure_ascii=False，但加上也不影响)
with open('result_NQ.json', 'w', encoding='utf-8') as f:
    json.dump(res_sum, f, ensure_ascii=False, indent=4)

print("数据已成功保存到 result.json 文件中！")