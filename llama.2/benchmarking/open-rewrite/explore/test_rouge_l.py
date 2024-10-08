from rouge_score import rouge_scorer
import pandas as pd

# Example data
generated_summaries = ["As a business professional, you know that productivity is key to success. However, the daily grind can be overwhelming, especially when it comes to the mundane tasks that often get in the way of our goals. In this article, we'll explore a step-by-step approach to managing the day's tasks, from phone calls to documents, and leave you with a clear plan for staying focused and achieving your objectives."]
reference_summaries = ["Typical office tasks may seem mundane, but they are vitally important to maintaining your business’ efficiency and productivity. Phone calls will have to be made, emails must be sent, and documents will always need to be signed. Finding an efficient way to deal with such tasks will prove crucial to the success of your business. Such banal yet necessary tasks often crop out outside of the office as well, but can be just as important to saving time and streamlining efficiency. You may have dishes to wash, or meals to prepare and clean up after, or any number of other chores that need to be accomplished throughout your day. The best way to handle such tasks is to assign set firm timeframes for accomplishing each task. This will ensure that you handle everything you need to in a timely manner, giving you more time to focus on running your business."]

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Initialize accumulators
total_precision = 0
total_recall = 0
total_f1 = 0
num_instances = len(generated_summaries)

# Calculate ROUGE-L for each instance
for gen, ref in zip(generated_summaries, reference_summaries):
    scores = scorer.score(ref, gen)
    rougeL = scores['rougeL']
    total_precision += rougeL.precision
    total_recall += rougeL.recall
    total_f1 += rougeL.fmeasure

# Compute micro-averaged scores
micro_avg_precision = total_precision / num_instances
micro_avg_recall = total_recall / num_instances
micro_avg_f1 = total_f1 / num_instances

# Print results
print(total_precision)

print(f"Micro-Averaged ROUGE-L Precision: {micro_avg_precision:.4f}")
print(f"Micro-Averaged ROUGE-L Recall: {micro_avg_recall:.4f}")
print(f"Micro-Averaged ROUGE-L F1 Score: {micro_avg_f1:.4f}")