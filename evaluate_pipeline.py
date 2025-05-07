import os
from langchain.chat_models import ChatOpenAI
from langchain.evaluation import load_evaluator
from langchain.schema import RunInfo, LLMResult
from langsmith import Client

# Load API keys from .env
from dotenv import load_dotenv
load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = Client(api_key=LANGSMITH_API_KEY)

# Define your test questions and ground truths
test_samples = [
    {
        "question": "What is a black hole?",
        "expected": "A black hole is a region in space with gravity so strong that not even light can escape."
    },
    {
        "question": "How do astronauts sleep in space?",
        "expected": "Astronauts sleep in sleeping bags attached to walls or ceilings to avoid floating around."
    },
    {
        "question": "What causes the phases of the Moon?",
        "expected": "The phases are caused by the changing angles of sunlight as the Moon orbits Earth."
    },
]

# Set up the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Load LangChain evaluators
accuracy_evaluator = load_evaluator("criteria", criteria="accuracy")
hallucination_evaluator = load_evaluator("criteria", criteria="hallucination")
context_evaluator = load_evaluator("criteria", criteria="relevance")

# Track results
results = []

for sample in test_samples:
    question = sample["question"]
    expected = sample["expected"]

    print(f"\n🔍 Evaluating question: {question}")
    
    # Get model answer
    answer = llm.predict(question)
    print(f"✅ Model answer: {answer}")
    print(f"🎯 Expected: {expected}")

    # Evaluate accuracy
    acc_score = accuracy_evaluator.evaluate_strings(prediction=answer, reference=expected)
    print(f"🧪 Accuracy: {acc_score['score']}")

    # Evaluate hallucination
    halluc_score = hallucination_evaluator.evaluate_strings(prediction=answer, reference=expected)
    print(f"🧪 Hallucination risk: {halluc_score['score']}")

    # Evaluate context relevance
    context_score = context_evaluator.evaluate_strings(prediction=answer, reference=expected)
    print(f"🧪 Context relevance: {context_score['score']}")

    # Log results to LangSmith (optional)
    run_info = RunInfo(run_id=f"eval-{question.replace(' ', '-')}")
    client.create_example(
        inputs={"question": question},
        outputs={"answer": answer},
        run_info=run_info,
        dataset_name="space-qa-eval"
    )

    results.append({
        "question": question,
        "answer": answer,
        "accuracy": acc_score['score'],
        "hallucination": halluc_score['score'],
        "context_relevance": context_score['score']
    })

# Summary
print("\n📊 Evaluation Summary:")
for res in results:
    print(f"- {res['question']}: Acc={res['accuracy']}, Halluc={res['hallucination']}, Relevance={res['context_relevance']}")
