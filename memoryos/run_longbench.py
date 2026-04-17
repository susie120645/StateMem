import sys
import os
import json
import random
import time
from datetime import datetime

# ✅ 1. 强制环境配置 (解决 Protobuf)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ✅ 2. 路径修正 (这是解决 ModuleNotFoundError 的关键)
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取 memoryos 目录 (package dir)
memoryos_pkg_dir = os.path.dirname(current_script_path)
# 获取项目根目录 (project root: memoryos-mcp)
project_root = os.path.dirname(memoryos_pkg_dir)

# 将项目根目录加入 sys.path 的第一位
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 再次确认 memoryos 包在路径中
try:
    from memoryos.memoryos import Memoryos
except ImportError:
    # 如果还是找不到，尝试直接从当前目录导入 (作为 fallback)
    sys.path.append(memoryos_pkg_dir)
    from memoryos import Memoryos

# ==============================================================================
# ⚙️ 实验配置 (LongBench / HotpotQA)
# ==============================================================================

# ⚠️ 请确保你已经上传了文件，并修改下面的路径！
# 如果你还没有文件，脚本会自动生成模拟数据
DATA_PATH = "/masiqi/code/jsr/MemoryOS/data/hotpotqa.jsonl" 
OLLAMA_BASE = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"

def generate_mock_longbench_data():
    """如果在本地找不到数据集，生成模拟数据以跑通流程"""
    print("⚠️ Dataset not found locally. Generating MOCK LongBench data...")
    mock_data = []
    for i in range(5):
        # 模拟长文本
        context = f"This is a long document about Physics. " * 50 + \
                  f"Key fact: The answer to query {i} is '42'. " + \
                  f"More random text... " * 50
        mock_data.append({
            "input": f"What is the answer to query {i}?",
            "context": context,
            "answers": ["42"],
            "length": len(context)
        })
    return mock_data

def run_longbench_eval():
    print("==================================================================")
    print("   A-MEM Benchmark: LongBench (HotpotQA) Evaluation")
    print("   Target: Test Multi-Hop Retrieval in Long Context")
    print("==================================================================\n")

    # 1. 加载数据
    dataset = []
    if os.path.exists(DATA_PATH):
        print(f"📂 Loading real dataset from {DATA_PATH}...")
        try:
            with open(DATA_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset.append(json.loads(line))
            # 随机抽样 10 条做快速测试 (全量太慢)
            dataset = random.sample(dataset, min(10, len(dataset)))
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            dataset = generate_mock_longbench_data()
    else:
        dataset = generate_mock_longbench_data()
    
    results = []
    
    # 2. 开始评测
    for idx, item in enumerate(dataset):
        print(f"\n🔹 [Case {idx+1}/{len(dataset)}] Processing...")
        
        # 为每个 Case 创建独立用户，清空记忆
        user_id = f"lb_user_{int(time.time())}_{idx}"
        data_storage = os.path.join(project_root, "memoryos_data")
        
        memo = Memoryos(
            user_id=user_id, 
            openai_api_key="ollama",
            openai_base_url=OLLAMA_BASE, 
            data_storage_path=data_storage,
            llm_model=LLM_MODEL,
            embedding_model_name=EMBED_MODEL,
            short_term_capacity=5
        )
        
        context = item['context']
        question = item['input']
        answers = item['answers']
        
        # 3. 注入长文本 (Simulate Streaming Ingestion)
        # 将长文本切分为多个 Chunk，模拟日志流，测试 G0 过滤效果
        print(f"   -> Injecting context ({len(context)} chars)...")
        # 简单切分，每 500 字符一段
        chunks = [context[i:i+500] for i in range(0, len(context), 500)]
        
        for i, chunk in enumerate(chunks):
            # 模拟注入
            if i % 5 == 0: sys.stdout.write("."); sys.stdout.flush()
            memo.add_memory(user_input=chunk, agent_response="Processed.")
        print(" Done.")
        
        # 4. 强制固化 (Consolidation)
        print("   -> Consolidating memory...")
        retry = 0
        while len(memo.short_term_memory.get_all()) > 0 and retry < 5:
            memo.updater.process_short_term_to_mid_term()
            retry += 1
            
        # 5. 检索回答
        print(f"   -> 🔍 Querying: '{question}'")
        try:
            response = memo.get_response(question)
        except Exception as e:
            response = ""
            print(f"   ❌ Error: {e}")
        
        # 6. 判定正确性 (只要包含任一标准答案就算对)
        is_correct = any(ans.lower() in response.lower() for ans in answers)
        mark = "✅" if is_correct else "❌"
        
        # 7. 统计存储量 (Storage Cost)
        stored_items = 0
        try:
            if os.path.exists(memo.mid_term_memory.file_path):
                with open(memo.mid_term_memory.file_path, 'r') as f:
                    data = json.load(f)
                    for s in data.get('sessions', {}).values():
                        stored_items += len(s.get('details', []))
        except: pass
        
        # 计算压缩率
        input_chunks = len(chunks)
        compression = (stored_items / input_chunks) * 100 if input_chunks > 0 else 0
        
        print(f"   -> Result: {mark} | Ans: {answers[0]}... | Resp: {response[:30]}... | Storage: {compression:.1f}%")
        
        results.append({
            "question": question,
            "correct": is_correct,
            "compression": compression
        })

    # 8. 最终统计
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    avg_compression = sum(r['compression'] for r in results) / len(results)
    
    print("\n" + "="*80)
    print(f"   🏆 LongBench Result Summary")
    print("="*80)
    print(f"   Accuracy:    {accuracy*100:.2f}%")
    print(f"   Storage:     {avg_compression:.2f}% (vs 100% baseline)")
    print("="*80)

if __name__ == "__main__":
    run_longbench_eval()