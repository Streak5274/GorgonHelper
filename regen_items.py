"""Regenerate pg_items.json from Originals/items.json.
New format: id -> [name, maxStack, iconId, value, keywords]
  - value: integer coin value (0 if missing)
  - keywords: list of keyword strings (may include quality like "VegetarianDish=84")
"""
import json, re

ORIG = 'Originals/items.json'
OUT  = 'pg_items.json'

with open(ORIG, encoding='utf-8') as f:
    orig = json.load(f)

result = {}
for key, item in orig.items():
    m = re.match(r'item_(\d+)', key)
    if not m:
        continue
    item_id = int(m.group(1))
    name      = item.get('Name', '')
    max_stack = item.get('MaxStackSize', 1)
    icon_id   = item.get('IconId', 0)
    value     = item.get('Value', 0)
    keywords  = item.get('Keywords', [])
    result[item_id] = [name, max_stack, icon_id, value, keywords]

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(result, f, separators=(',', ':'))

print(f"Written {len(result)} items to {OUT}")
# Quick spot check
for tid in [6008, 1]:
    if tid in result:
        print(f"  item {tid}: {result[tid]}")
