import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import asyncio
from api_server import inspect_unit, find_image, INDUSTRIAL_SUITE

async def main():
    print("Testing all 8 production lines with curated samples...")
    for item in INDUSTRIAL_SUITE:
        cat = item["id"]
        good_path = str(find_image(cat, "good", 0))
        defect_path = str(find_image(cat, "defect", 0))

        res_good = await inspect_unit(category=cat, target_name="sample_good.png", sample_path=good_path)
        res_defect = await inspect_unit(category=cat, target_name="sample_defect.png", sample_path=defect_path)

        good_ok = (res_good["verdict"] in ["NORMAL_PASS", "NOMINAL_PASS"])
        defect_ok = (res_defect["verdict"] == "CRITICAL_DEFECT")
        status = "PASSED" if (good_ok and defect_ok) else "FAILED"

        print(f"[{status}] Line: {cat:10s} | Normal: {res_good['verdict']} (score: {res_good['score']:.2f}, thresh: {res_good['threshold']:.2f}) | Defect: {res_defect['verdict']} (score: {res_defect['score']:.2f}, thresh: {res_defect['threshold']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())
