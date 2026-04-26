#!/usr/bin/env python3
import json
import os

in_file = "data/generator/train.jsonl"
out_file = "data/generator/train_zh_dummy.jsonl"

print(f"I2: 正在准备临时测试数据 (截取前 1000 条)...\n来源: {in_file}\n输出: {out_file}")

count = 0
with open(in_file, 'r', encoding='utf-8') as f_in, open(out_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        # 这里可以直接插入一些翻译逻辑或真实数据替换
        # 目前为了验证 Pipeline，直接采样前 1000 条
        data = json.loads(line)
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
        count += 1
        if count >= 1000:
            break

print(f"✅ 成功生成 {count} 条测试数据！后续请替换为真实中文数据。")
