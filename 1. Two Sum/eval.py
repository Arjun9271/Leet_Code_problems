import os
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from huggingface_hub import InferenceClient
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from datasets import load_dataset
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

class LLMEvaluator:
    """A comprehensive framework for evaluating LLMs on long context tasks."""
    
    def __init__(self, api_key, models_config, max_workers=3):
        """
        Initialize the evaluator with API key and model configurations.
        
        Args:
            api_key: HuggingFace API key
            models_config: List of model configurations with provider and model name
            max_workers: Maximum number of concurrent evaluation threads
        """
        self.api_key = api_key
        self.models_config = models_config
        self.max_workers = max_workers
        self.clients = {}
        
        # Initialize clients for each model
        for config in self.models_config:
            client = InferenceClient(
                provider=config["provider"],
                api_key=self.api_key
            )
            self.clients[config["model"]] = client
        
        # Evaluation metrics
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method1
        
        # Results storage
        self.results = {}
        
    def generate_response(self, model, prompt, context, temperature=0.1, max_tokens=1024, timeout=120):
        """
        Generate a response from the specified model.
        
        Args:
            model: The model name to use
            prompt: The prompt text to send to the model
            context: The context document text
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds
            
        Returns:
            Generated response text
        """
        try:
            client = self.clients[model]
            
            # Start timing
            start_time = time.time()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "text",
                            "text": context
                        }
                    ]
                }
            ]
            
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Collect response content
            response_content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    response_content += chunk.choices[0].delta.content
                    
                # Check timeout
                if time.time() - start_time > timeout:
                    print(f"Response generation timed out for {model}")
                    break
            
            # Calculate response time
            response_time = time.time() - start_time
            
            return {
                "response": response_content.strip(),
                "response_time": response_time
            }
            
        except Exception as e:
            print(f"Error generating response from {model}: {str(e)}")
            return {
                "response": f"ERROR: {str(e)}",
                "response_time": -1
            }
            
    def evaluate_factual_recall(self, response, ground_truth):
        """
        Evaluate the factual recall of a response against ground truth.
        
        Args:
            response: The model's response text
            ground_truth: The ground truth text
            
        Returns:
            Dictionary of evaluation metrics
        """
        # ROUGE scores
        rouge_scores = self.scorer.score(ground_truth, response)
        
        # BLEU score (with smoothing for shorter responses)
        response_tokens = nltk.word_tokenize(response.lower())
        ground_truth_tokens = nltk.word_tokenize(ground_truth.lower())
        bleu_score = sentence_bleu([ground_truth_tokens], response_tokens, 
                                  smoothing_function=self.smoothie)
        
        # Simple fact extraction (using regex for key information patterns)
        # This is a basic implementation and should be enhanced for specific tasks
        facts_found = 0
        facts_total = 0
        
        # Extract potential facts from ground truth (dates, numbers, proper names)
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}-\d{1,2}-\d{2,4}\b|\b\w+ \d{1,2}, \d{4}\b'
        number_pattern = r'\b\d+\.\d+\b|\b\d+\b'
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        
        ground_truth_dates = set(re.findall(date_pattern, ground_truth))
        ground_truth_numbers = set(re.findall(number_pattern, ground_truth))
        ground_truth_names = set(re.findall(name_pattern, ground_truth))
        
        response_dates = set(re.findall(date_pattern, response))
        response_numbers = set(re.findall(number_pattern, response))
        response_names = set(re.findall(name_pattern, response))
        
        # Count matched facts
        facts_total += len(ground_truth_dates) + len(ground_truth_numbers) + len(ground_truth_names)
        facts_found += len(ground_truth_dates.intersection(response_dates))
        facts_found += len(ground_truth_numbers.intersection(response_numbers))
        facts_found += len(ground_truth_names.intersection(response_names))
        
        fact_recall = facts_found / max(facts_total, 1)
        
        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            "bleu": bleu_score,
            "fact_recall": fact_recall
        }
        
    def evaluate_hallucination(self, response, context):
        """
        Evaluate the level of hallucination in a response.
        
        Args:
            response: The model's response text
            context: The context document text
            
        Returns:
            Hallucination score (higher means more likely hallucination)
        """
        # This is a simplified approach - could be enhanced with more sophisticated methods
        
        # Extract potential facts from response
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}-\d{1,2}-\d{2,4}\b|\b\w+ \d{1,2}, \d{4}\b'
        number_pattern = r'\b\d+\.\d+\b|\b\d+\b'
        name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        
        response_dates = set(re.findall(date_pattern, response))
        response_numbers = set(re.findall(number_pattern, response))
        response_names = set(re.findall(name_pattern, response))
        
        context_dates = set(re.findall(date_pattern, context))
        context_numbers = set(re.findall(number_pattern, context))
        context_names = set(re.findall(name_pattern, context))
        
        # Count facts in response not found in context
        hallucinated_facts = 0
        total_facts = len(response_dates) + len(response_numbers) + len(response_names)
        
        hallucinated_facts += len(response_dates.difference(context_dates))
        hallucinated_facts += len(response_numbers.difference(context_numbers))
        hallucinated_facts += len(response_names.difference(context_names))
        
        # Calculate hallucination score (percentage of facts not in context)
        if total_facts > 0:
            hallucination_score = hallucinated_facts / total_facts
        else:
            hallucination_score = 0.0
            
        return hallucination_score
        
    def run_benchmark(self, test_cases, verbose=True):
        """
        Run benchmark evaluation on multiple test cases.
        
        Args:
            test_cases: List of test case dictionaries with prompts, contexts, and ground truths
            verbose: Whether to print progress
            
        Returns:
            DataFrame of evaluation results
        """
        all_results = []
        
        # Process test cases with thread pool for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {}
            
            # Submit all tasks
            for test_idx, test_case in enumerate(test_cases):
                for model_config in self.models_config:
                    model_name = model_config["model"]
                    task = (test_idx, model_name, test_case)
                    future = executor.submit(self._evaluate_test_case, *task)
                    future_to_task[future] = (test_idx, model_name)
            
            # Process results as they complete
            with tqdm(total=len(future_to_task), desc="Evaluating", disable=not verbose) as pbar:
                for future in as_completed(future_to_task):
                    test_idx, model_name = future_to_task[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        print(f"Error evaluating test {test_idx} on {model_name}: {str(e)}")
                    pbar.update(1)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        self.results = results_df
        
        # Print summary if verbose
        if verbose:
            print("\nEvaluation Results Summary:")
            # Group by 'model' on the original DataFrame
            grouped_results = results_df.groupby('model')
            
            # Calculate the mean of numeric columns for each group
            summary = grouped_results.mean(numeric_only=True).round(4)
            print(summary)
            
        return results_df
        
    def _evaluate_test_case(self, test_idx, model_name, test_case):
        """Helper method to evaluate a single test case on a single model."""
        prompt = test_case["prompt"]
        context = test_case["context"]
        ground_truth = test_case.get("ground_truth", "")
        
        # Generate response
        response_data = self.generate_response(
            model=model_name,
            prompt=prompt,
            context=context
        )
        
        response = response_data["response"]
        response_time = response_data["response_time"]
        
        # Calculate metrics
        content_length = len(response.split())
        context_length = len(context.split())
        
        # Factual recall metrics
        recall_metrics = self.evaluate_factual_recall(response, ground_truth) if ground_truth else {}
        
        # Hallucination score
        hallucination_score = self.evaluate_hallucination(response, context)
        
        # Record result
        result = {
            "test_id": test_idx,
            "model": model_name,
            "context_length": context_length,
            "response_length": content_length,
            "response_time": response_time,
            "hallucination_score": hallucination_score,
            "response": response
        }
        
        # Add recall metrics if ground truth was provided
        result.update(recall_metrics)
        
        return result
    
    def visualize_results(self):
        """
        Visualize the evaluation results with charts.
        
        Returns:
            None (displays plots)
        """
        if self.results.empty:
            print("No results to visualize. Run benchmark first.")
            return
            
        # Set up the plotting area
        plt.figure(figsize=(18, 12))
        
        # 1. Response Time vs Context Length
        plt.subplot(2, 2, 1)
        sns.scatterplot(
            data=self.results, 
            x="context_length", 
            y="response_time", 
            hue="model",
            s=100,
            alpha=0.7
        )
        plt.title("Response Time vs Context Length")
        plt.xlabel("Context Length (tokens)")
        plt.ylabel("Response Time (seconds)")
        
        # 2. Response Length Distribution
        plt.subplot(2, 2, 2)
        sns.boxplot(
            data=self.results,
            x="model",
            y="response_length"
        )
        plt.title("Response Length Distribution")
        plt.xlabel("Model")
        plt.ylabel("Response Length (tokens)")
        
        # 3. ROUGE-L Scores (if available)
        if "rougeL" in self.results.columns:
            plt.subplot(2, 2, 3)
            sns.barplot(
                data=self.results,
                x="model",
                y="rougeL"
            )
            plt.title("ROUGE-L Scores")
            plt.xlabel("Model")
            plt.ylabel("ROUGE-L F1 Score")
            plt.ylim(0, 1)
        
        # 4. Hallucination Score
        plt.subplot(2, 2, 4)
        sns.barplot(
            data=self.results,
            x="model",
            y="hallucination_score"
        )
        plt.title("Hallucination Score")
        plt.xlabel("Model")
        plt.ylabel("Hallucination Score (lower is better)")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        # Detailed metrics comparison
        if "rouge1" in self.results.columns:
            plt.figure(figsize=(14, 8))
            
            metrics = ["rouge1", "rouge2", "rougeL", "bleu", "fact_recall"]
            metric_names = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU", "Fact Recall"]
            
            # Calculate average metrics per model
            metrics_df = self.results.groupby('model')[metrics].mean().reset_index()
            
            # Reshape for seaborn
            metrics_long = pd.melt(
                metrics_df, 
                id_vars=['model'], 
                value_vars=metrics,
                var_name='metric', 
                value_name='score'
            )
            
            # Map metric names
            metric_map = dict(zip(metrics, metric_names))
            metrics_long['metric'] = metrics_long['metric'].map(metric_map)
            
            # Plot
            sns.barplot(
                data=metrics_long,
                x='metric',
                y='score',
                hue='model'
            )
            
            plt.title("Comparison of Quality Metrics")
            plt.xlabel("Metric")
            plt.ylabel("Score")
            plt.ylim(0, 1)
            plt.legend(title="Model")
            plt.tight_layout()
            plt.show()
    
    def get_detailed_report(self):
        """
        Generate a detailed text report of evaluation results.
        
        Returns:
            String containing detailed report
        """
        if self.results.empty:
            return "No results to report. Run benchmark first."
            
        report = []
        report.append("=" * 80)
        report.append("DETAILED LLM EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        overall_metrics = self.results.groupby('model').mean().round(4)
        report.append("OVERALL METRICS:")
        report.append(overall_metrics.to_string())
        report.append("")
        
        # Performance by context length
        context_bins = [0, 1000, 5000, 10000, float('inf')]
        context_labels = ['<1K', '1K-5K', '5K-10K', '>10K']
        
        self.results['context_bin'] = pd.cut(
            self.results['context_length'],
            bins=context_bins,
            labels=context_labels
        )
        
        context_metrics = self.results.groupby(['model', 'context_bin']).mean().round(4)
        report.append("PERFORMANCE BY CONTEXT LENGTH:")
        report.append(context_metrics.to_string())
        report.append("")
        
        # Sample responses
        report.append("SAMPLE RESPONSES (First 3 test cases):")
        for test_id in sorted(self.results['test_id'].unique())[:3]:
            test_results = self.results[self.results['test_id'] == test_id]
            report.append(f"\nTest Case {test_id}:")
            
            for _, row in test_results.iterrows():
                model = row['model']
                response = row['response']
                
                # Truncate long responses for the report
                if len(response) > 500:
                    response = response[:500] + "... [truncated]"
                
                report.append(f"\n{model}:")
                report.append(response)
                report.append("-" * 40)
        
        return "\n".join(report)
        
    def export_results(self, filepath=None):
        """
        Export evaluation results to CSV.
        
        Args:
            filepath: Path to save CSV file (default: results_{timestamp}.csv)
            
        Returns:
            Path to saved file
        """
        if self.results.empty:
            print("No results to export. Run benchmark first.")
            return None
            
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"results_{timestamp}.csv"
            
        # Create a copy of results with truncated responses
        export_df = self.results.copy()
        export_df['response'] = export_df['response'].apply(
            lambda x: (x[:500] + "... [truncated]") if len(x) > 500 else x
        )
        
        export_df.to_csv(filepath, index=False)
        print(f"Results exported to {filepath}")
        return filepath

    @staticmethod
    def load_legal_contract_dataset(num_samples=5):
        """
        Load legal contract text dataset for evaluation.
        
        Args:
            num_samples: Number of samples to load
            
        Returns:
            List of test cases
        """
        try:
            # Load the Pile dataset subset containing legal contracts
            # The Pile contains a variety of document types including legal documents
            dataset = load_dataset("NeelNanda/pile-10k", split="train")
            
            # Filter for documents likely to be contracts (approximate)
            contract_texts = []
            for item in dataset:
                text = item['text']
                # Look for common contract terms
                if (("AGREEMENT" in text.upper() or "CONTRACT" in text.upper()) and 
                    ("PARTIES" in text.upper() or "WHEREAS" in text.upper() or "TERMS" in text.upper())):
                    contract_texts.append(text)
                    if len(contract_texts) >= num_samples * 2:  # Get extra for filtering
                        break
            
            # Select contracts that are reasonably long (have context)
            contract_texts = [text for text in contract_texts if len(text.split()) >= 1000]
            
            # Take a sample
            if len(contract_texts) > num_samples:
                contract_texts = random.sample(contract_texts, num_samples)
            
            test_cases = []
            for i, text in enumerate(contract_texts):
               
                test_case = {
                    "prompt": (
                        "Extract the following information from this legal contract:\n"
                        "1. Who are the parties to this agreement?\n"
                        "2. What is the effective date of the agreement?\n"
                        "3. What is the term or duration of the agreement?\n"
                        "4. What are the key obligations of each party?\n"
                        "5. Are there any important clauses regarding termination?\n"
                        "Please be specific and cite relevant sections when possible."
                    ),
                    "context": text,
                    "ground_truth": ""  # No ground truth for this dataset
                }
                test_cases.append(test_case)
            
            return test_cases
            
        except Exception as e:
            print(f"Error loading legal contract dataset: {str(e)}")
            # Fallback to synthetic test cases
            return LLMEvaluator.create_synthetic_test_cases(num_samples)
    
    @staticmethod
    def load_scientific_paper_dataset(num_samples=5):
       
        try:
            # Load scientific papers from ArXiv dataset
            dataset = load_dataset("arxiv_dataset", split="train")
            
            # Select random papers
            papers = dataset.select(range(min(100, len(dataset))))
            papers = random.sample(papers, min(num_samples, len(papers)))
            
            test_cases = []
            for paper in papers:
                abstract = paper.get('abstract', '')
                intro = paper.get('introduction', '')
                
                # Combine abstract and intro for context
                context = abstract + "\n\n" + intro
                
                # Create test case with summary and analysis questions
                test_case = {
                    "prompt": (
                        "Analyze this scientific paper excerpt and answer the following questions:\n"
                        "1. What is the main research question or objective?\n"
                        "2. What methodology is being used?\n"
                        "3. What are the key findings or contributions?\n"
                        "4. What are the limitations of this research?\n"
                        "5. What future work is suggested?\n"
                        "Provide a comprehensive analysis based solely on the provided text."
                    ),
                    "context": context,
                    "ground_truth": ""  # No ground truth for this dataset
                }
                test_cases.append(test_case)
            
            return test_cases
            
        except Exception as e:
            print(f"Error loading scientific paper dataset: {str(e)}")
            # Fallback to synthetic test cases
            return LLMEvaluator.create_synthetic_test_cases(num_samples)
    
    @staticmethod
    def create_synthetic_test_cases(num_cases=5):
      
        # Template for a synthetic legal contract
        legal_template = """
        VARIATION AGREEMENT
        
        THIS VARIATION AGREEMENT (the "Variation") is made on {agreement_date}
        
        BETWEEN:
        
        (1) BRITISH TELECOM PLC, a company registered in England and Wales (company number {bt_reg_number}) whose registered office is at 81 Newgate Street, London, EC1A 7AJ ("BT"); and
        
        (2) {company_name}, a company registered in {company_country} (company number {company_reg_number}) whose registered office is at {company_address} (the "Supplier").
        
        BACKGROUND:
        
        (A) BT and the Supplier are parties to an agreement dated {original_agreement_date} (the "Agreement") for the provision by the Supplier to BT of {service_description}.
        
        (B) The parties wish to vary the Agreement as set out in this Variation.
        
        IT IS AGREED as follows:
        
        1. INTERPRETATION
        
        1.1 In this Variation, capitalised terms shall have the same meaning as in the Agreement, unless defined otherwise in this Variation.
        
        1.2 References to clauses are to clauses of the Agreement, unless stated otherwise.
        
        2. VARIATION
        
        2.1 With effect from {effective_date} (the "Variation Date"), the Agreement shall be varied as follows:
        
        2.1.1 Clause {clause_number} of the Agreement shall be deleted and replaced with:
        "{new_clause_text}"
        
        2.1.2 Clause {second_clause_number} shall be amended by replacing the text "{old_text}" with "{new_text}".
        
        2.1.3 The following new clause shall be inserted after clause {insert_after_clause}:
        "{new_inserted_clause}"
        
        3. GENERAL
        
        3.1 This Variation may be executed in any number of counterparts, each of which when executed shall constitute a duplicate original, but all the counterparts shall together constitute the one Variation.
        
        3.2 Except as varied by this Variation, the Agreement shall remain in full force and effect.
        
        3.3 This Variation shall be governed by and construed in accordance with English law.
        
        SIGNED for and on behalf of BRITISH TELECOM PLC:
        
        By: {bt_signatory_name}
        Position: {bt_signatory_position}
        Date: {bt_signing_date}
        
        SIGNED for and on behalf of {company_name}:
        
        By: {supplier_signatory_name}
        Position: {supplier_signatory_position}
        E-signed date: {esigned_date}
        
        SCHEDULE 1
        
        AMENDED PRICING SCHEDULE
        
        1. Pricing Table
        
        Service | Previous Price | New Price | % Change
        --------|----------------|-----------|----------
        {service_1} | £{prev_price_1} | £{new_price_1} | {change_percent_1}%
        {service_2} | £{prev_price_2} | £{new_price_2} | {change_percent_2}%
        {service_3} | £{prev_price_3} | £{new_price_3} | {change_percent_3}%
        
        2. Payment Terms
        
        2.1 Invoicing shall be {invoicing_frequency}.
        
        2.2 Payment terms are {payment_terms} from the date of invoice.
        
        3. Duration
        
        3.1 The initial term shall be {initial_term} years from the Variation Date (the "Initial Term").
        
        3.2 Following the Initial Term, the Agreement shall automatically renew for successive periods of {renewal_term} year(s) (each a "Renewal Term") unless either party gives written notice to terminate at least {notice_period} months before the end of the Initial Term or the relevant Renewal Term.
        
        4. Qualifying Periods
        
        4.1 Year1 qualifying period: {year1_period}
        4.2 Year2 qualifying period: {year2_period}
        4.3 Year3 qualifying period: {year3_period}
        
        5. Contact Information
        
        5.1 BT Primary Contact:
        Name: {bt_contact_name}
        Email: {bt_contact_email}
        Phone: {bt_contact_phone}
        
        5.2 Supplier Primary Contact:
        Name: {supplier_contact_name}
        Email: {supplier_contact_email}
        Phone: {supplier_contact_phone}
        
        5.3 For urgent matters, the Director of {department_name}, {director_name}, should be contacted at {director_email}.
        
        APPENDIX A - STATEMENT OF WORK
        
        This Statement of Work ("SOW") is entered into pursuant to the Agreement between BT and the Supplier.
        
        Project Name: {project_name}
        SOW Reference: {sow_reference}
        SOW Start Date: {sow_start_date}
        SOW End Date: {sow_end_date}
        
        Project Manager:
        {project_manager_name}
        {project_manager_email}
        {project_manager_phone}
        
        Deliverables:
        1. {deliverable_1} - Due by {deliverable_1_date}
        2. {deliverable_2} - Due by {deliverable_2_date}
        3. {deliverable_3} - Due by {deliverable_3_date}
        
        Acceptance Criteria:
        {acceptance_criteria}
        
        Change Control Process:
        {change_control_process}
        
        APPENDIX B - SERVICE LEVEL AGREEMENT
        
        1. Service Availability
        
        Service | Target Availability | Measurement Period
        --------|---------------------|-------------------
        {sla_service_1} | {sla_target_1}% | {sla_period_1}
        {sla_service_2} | {sla_target_2}% | {sla_period_2}
        
        2. Service Credits
        
        Availability Level | Service Credit
        ------------------|---------------
        Below target but above {sla_level_1}% | {sla_credit_1}% of monthly charges
        Below {sla_level_1}% but above {sla_level_2}% | {sla_credit_2}% of monthly charges
        Below {sla_level_2}% | {sla_credit_3}% of monthly charges
        
        3. Reporting
        
        3.1 The Supplier shall provide service reports on a {reporting_frequency} basis.
        
        3.2 Reports shall be sent to {report_recipient_name} at {report_recipient_email}.
        
        Yan Arnauld
        Technical Delivery Manager
        yan.arnauld@bt.com
        """
        
        test_cases = []
        
        # Generate synthetic test cases with varied context lengths
        for i in range(num_cases):
            # Random values for template
            values = {
                "agreement_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                "bt_reg_number": f"{random.randint(1000000, 9999999)}",
                "company_name": f"Acme Services {random.choice(['Limited', 'Inc.', 'GmbH', 'SA'])}",
                "company_country": random.choice(["England and Wales", "United States", "Germany", "France"]),
                "company_reg_number": f"{random.randint(1000000, 9999999)}",
                "company_address": f"{random.randint(1, 100)} Business Avenue, {random.choice(['London', 'New York', 'Berlin', 'Paris'])}",
                "original_agreement_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2022",
                "service_description": "IT infrastructure services and support",
                "effective_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                "clause_number": f"{random.randint(1, 10)}.{random.randint(1, 5)}",
                "new_clause_text": "The Supplier shall provide the Services in accordance with the updated service levels set out in Schedule 1.",
                "second_clause_number": f"{random.randint(1, 10)}.{random.randint(1, 5)}",
                "old_text": "quarterly basis",
                "new_text": "monthly basis",
                "insert_after_clause": f"{random.randint(1, 10)}.{random.randint(1, 5)}",
                "new_inserted_clause": "The Supplier shall comply with BT's updated security policies as notified to the Supplier from time to time.",
                "bt_signatory_name": f"{random.choice(['John', 'Sarah', 'David', 'Emma'])} {random.choice(['Smith', 'Jones', 'Taylor', 'Brown'])}",
                "bt_signatory_position": random.choice(["Procurement Director", "Head of IT Services", "Commercial Director", "CTO"]),
                "bt_signing_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                "supplier_signatory_name": f"{random.choice(['Michael', 'Jennifer', 'Robert', 'Lisa'])} {random.choice(['Wilson', 'Moore', 'Clark', 'Lewis'])}",
                "supplier_signatory_position": random.choice(["CEO", "COO", "Managing Director", "Sales Director"]),
                "esigned_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                "service_1": "Cloud Infrastructure",
                "prev_price_1": f"{random.randint(10000, 50000)}",
                "new_price_1": f"{random.randint(10000, 50000)}",
                "change_percent_1": f"{random.randint(-10, 15)}",
                "service_2": "Network Services",
                "prev_price_2": f"{random.randint(5000, 20000)}",
                "new_price_2": f"{random.randint(5000, 20000)}",
                "change_percent_2": f"{random.randint(-10, 15)}",
                "service_3": "Managed Security",
                "prev_price_3": f"{random.randint(8000, 30000)}",
                "new_price_3": f"{random.randint(8000, 30000)}",
                "change_percent_3": f"{random.randint(-10, 15)}",
                "invoicing_frequency": random.choice(["monthly", "quarterly", "annually"]),
                "payment_terms": f"{random.choice(['30', '45', '60'])} days",
                "initial_term": random.randint(1, 5),
                "renewal_term": random.randint(1, 3),
                "notice_period": random.randint(1, 6),
                "year1_period": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023 - {random.randint(1, 28)}/{random.randint(1, 12)}/2024",
                "year2_period": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2024 - {random.randint(1, 28)}/{random.randint(1, 12)}/2025",
                "year3_period": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2025 - {random.randint(1, 28)}/{random.randint(1, 12)}/2026",
                "bt_contact_name": f"{random.choice(['James', 'Emily', 'Andrew', 'Sophie'])} {random.choice(['Williams', 'Davis', 'Martin', 'Thompson'])}",
                "bt_contact_email": f"bt.contact{random.randint(1, 999)}@bt.com",
                "bt_contact_phone": f"+44 {random.randint(1000000000, 9999999999)}",
                "supplier_contact_name": f"{random.choice(['Thomas', 'Rebecca', 'Steven', 'Amanda'])} {random.choice(['Johnson', 'Miller', 'Wilson', 'Davis'])}",
                "supplier_contact_email": f"supplier.contact{random.randint(1, 999)}@acme.com",
                "supplier_contact_phone": f"+{random.randint(1, 9)}{random.randint(10000000000, 99999999999)}",
                "department_name": random.choice(["IT Services", "Procurement", "Operations", "Technology"]),
                "director_name": "Yan Arnauld",
                "director_email": "yan.arnauld@bt.com",
                "project_name": f"Project {random.choice(['Alpha', 'Beta', 'Gamma', 'Delta'])}",
                "sow_reference": f"SOW-{random.randint(1000, 9999)}-{random.randint(100, 999)}",
                "sow_start_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                "sow_end_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2024",
                "project_manager_name": f"{random.choice(['Patrick', 'Laura', 'Kevin', 'Rachel'])} {random.choice(['Roberts', 'Turner', 'Phillips', 'Campbell'])}",
                "project_manager_email": f"pm{random.randint(1, 999)}@acme.com",
                "project_manager_phone": f"+{random.randint(1, 9)}{random.randint(10000000000, 99999999999)}",
                "deliverable_1": "Initial Assessment Report",
                "deliverable_1_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                "deliverable_2": "Implementation Plan",
                "deliverable_2_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2023",
                "deliverable_3": "Final Deployment",
                "deliverable_3_date": f"{random.randint(1, 28)}/{random.randint(1, 12)}/2024",
                "acceptance_criteria": "All deliverables must be approved by BT's Project Manager within 10 business days of submission.",
                "change_control_process": "All changes must be documented in writing and approved by both parties before implementation.",
                "sla_service_1": "Cloud Infrastructure",
                "sla_target_1": f"{random.randint(95, 99)}.{random.randint(0, 9)}",
                "sla_period_1": "Monthly",
                "sla_service_2": "Network Services",
                "sla_target_2": f"{random.randint(95, 99)}.{random.randint(0, 9)}",
                "sla_period_2": "Monthly",
                "sla_level_1": f"{random.randint(90, 95)}",
                "sla_credit_1": f"{random.randint(5, 15)}",
                "sla_level_2": f"{random.randint(85, 90)}",
                "sla_credit_2": f"{random.randint(15, 30)}",
                "sla_credit_3": f"{random.randint(30, 50)}",
                "reporting_frequency": random.choice(["weekly", "monthly", "quarterly"]),
                "report_recipient_name": f"{random.choice(['Mark', 'Karen', 'Daniel', 'Victoria'])} {random.choice(['White', 'Green', 'Baker', 'Hill'])}",
                "report_recipient_email": f"reports{random.randint(1, 999)}@bt.com"
            }
            
            # Fill the template with random values
            contract_text = legal_template.format(**values)
            
            # Create a test case with a question prompt
            questions = [
                "What is the E-signed date mentioned in the document?",
                "Who is the director mentioned in the document?",
                "When was the variation agreement dated?",
                "Where has British Telecom PLC been registered?",
                "What is the Year3 qualifying period?",
                "What is the email of Yan Arnauld mentioned in the document?"
            ]
            
            # Create ground truth based on generated values
            ground_truth = f"""
            1. The E-signed date is {values['esigned_date']}
            2. The director mentioned is {values['director_name']}
            3. The variation agreement was dated on {values['agreement_date']}
            4. British Telecom PLC is registered in England and Wales
            5. The Year3 qualifying period is {values['year3_period']}
            6. Yan Arnauld's email is {values['director_email']}
            """
            
            # Create test case
            test_case = {
                "prompt": "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)]),
                "context": contract_text,
                "ground_truth": ground_truth
            }
            
            test_cases.append(test_case)
        
        return test_cases

    @staticmethod
    def create_custom_test_case(prompt, context, ground_truth=""):
        """
        Create a custom test case with provided prompt, context, and optional ground truth.
        
        Args:
            prompt: The question or instruction to the model
            context: The document context
            ground_truth: Optional ground truth answer
            
        Returns:
            Test case dictionary
        """
        return {
            "prompt": prompt,
            "context": context,
            "ground_truth": ground_truth
        }

# Example usage

def run_llm_evaluation_demo():
    """
    Demonstration of using the LLM Evaluator framework.
    """
    # Replace with your actual API key
    api_key = "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"
    
    # Define models to evaluate
    models_config = [
        {
            "provider": "novita",
            "model": "meta-llama/Llama-3.2-3B-Instruct"
        },
        {
            "provider": "fireworks-ai",
            "model": "meta-llama/Llama-3.1-8B-Instruct"
        }
    ]
    
    # Initialize evaluator
    evaluator = LLMEvaluator(api_key, models_config)
    
    # Create test cases (either from datasets or synthetic)
    # Using synthetic for demonstration
    print("Creating test cases...")
    test_cases = LLMEvaluator.create_synthetic_test_cases(num_cases=3)
    
    # Optional: Add some scientific paper test cases too
    try:
        print("Adding scientific paper test cases...")
        paper_test_cases = LLMEvaluator.load_scientific_paper_dataset(num_samples=2)
        test_cases.extend(paper_test_cases)
    except:
        print("Could not load scientific paper test cases, using more synthetic ones.")
        test_cases.extend(LLMEvaluator.create_synthetic_test_cases(num_cases=2))
    
    # Run the benchmark
    print("Running benchmark evaluation...")
    results = evaluator.run_benchmark(test_cases)
    
    # Generate and print detailed report
    print("\nGenerating detailed report...")
    report = evaluator.get_detailed_report()
    print(report)
    
    # Visualize results
    print("\nVisualizing results...")
    evaluator.visualize_results()
    
    # Export results to CSV
    evaluator.export_results()
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    run_llm_evaluation_demo()
