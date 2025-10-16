import jsonlines

# 读取“longbench.jsonl”文件

tps = set()
with jsonlines.open('longbench.jsonl') as reader:
    for obj in reader:
        tps.add(obj['type'])
        
print(tps)