"""
Org Chart Evaluation Script

Evaluates a generated org chart against ground truth data by:
1. Fuzzy name matching (1-to-1)
2. Coverage metrics
3. Manager relationship accuracy

Usage:
    python evaluate_org_chart.py --pred <generated_file> --true <ground_truth_file>
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse
import re

from rapidfuzz import fuzz, process

from orgchart.model import OrgChart


@dataclass
class EvaluationResults:
    name_mapping: Dict[str, str]  # ground_truth_name -> predicted_name
    unmatched_gt: List[str]
    unmatched_pred: List[str]
    coverage_pct: float
    manager_accuracy: float
    manager_correct: int
    manager_total: int
    manager_errors: List[Dict]


class OrgChartEvaluator:
    def __init__(self, predicted: OrgChart, ground_truth: OrgChart):
        self.predicted = predicted
        self.ground_truth = ground_truth
        # Create lookup dictionaries for easy access by name
        self.gt_lookup = {entry.name: entry.model_dump() for entry in ground_truth.entries}
        self.pred_lookup = {entry.name: entry.model_dump() for entry in predicted.entries}

    def fuzzy_match_names(
        self, threshold: int = 80
    ) -> Tuple[Dict[str, str], List[str], List[str]]:
        """
        Match ground truth names to predicted names using fuzzy matching.
        Ensures 1-to-1 matching.

        Args:
            threshold: Minimum similarity score (0-100) to consider a match

        Returns:
            Tuple of (name_mapping, unmatched_gt, unmatched_pred)
        """
        gt_names = list(self.gt_lookup.keys())
        pred_names = list(self.pred_lookup.keys())

        # Store all potential matches with scores
        potential_matches = []

        for gt_name in gt_names:
            # Normalize the ground truth name for matching
            gt_name_normalized = _normalize_name(gt_name)

            # Try exact substring match first (e.g., "Vic" in "Victor Zhou")
            substring_matches = [
                p
                for p in pred_names
                if gt_name_normalized.startswith(_normalize_name(p))
                or _normalize_name(p).startswith(gt_name_normalized)
            ]
            for match in substring_matches:
                # Perfect substring match
                potential_matches.append((gt_name, match, threshold))
            
            # Use fuzzy matching with normalized name
            matches = process.extract(
                gt_name_normalized,
                pred_names,
                scorer=fuzz.token_sort_ratio,
                limit=3,
            )
            for match_name, score, _ in matches:
                if score >= threshold:
                    potential_matches.append((gt_name, match_name, score))

        # Sort by score (highest first) to prioritize best matches
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        for match in potential_matches:
            print(match)
        
        # Greedy 1-to-1 matching
        name_mapping = {}
        used_pred_names = set()

        for gt_name, pred_name, score in potential_matches:
            if gt_name not in name_mapping and pred_name not in used_pred_names:
                name_mapping[gt_name] = pred_name
                used_pred_names.add(pred_name)

        # Find unmatched names
        unmatched_gt = [name for name in gt_names if name not in name_mapping]
        unmatched_pred = [name for name in pred_names if name not in used_pred_names]

        return name_mapping, unmatched_gt, unmatched_pred

    def evaluate_coverage(
        self,
        name_mapping: Dict[str, str],
        unmatched_gt: List[str],
        unmatched_pred: List[str],
    ) -> Dict:
        """
        Evaluate coverage metrics.

        Returns:
            Dictionary with coverage statistics
        """
        total_gt = len(self.gt_lookup)
        matched_count = len(name_mapping)

        coverage_pct = (matched_count / total_gt * 100) if total_gt > 0 else 0

        return {
            "total_ground_truth": total_gt,
            "total_predicted": len(self.pred_lookup),
            "matched": matched_count,
            "coverage_pct": coverage_pct,
            "missing_from_predicted": unmatched_gt,
            "extra_in_predicted": unmatched_pred,
        }

    def evaluate_managers(self, name_mapping: Dict[str, str]) -> Dict:
        """
        Evaluate manager relationship accuracy.

        Returns:
            Dictionary with manager accuracy metrics
        """
        correct = 0
        total = 0
        errors = []

        for gt_name, pred_name in name_mapping.items():
            gt_person = self.gt_lookup[gt_name]
            pred_person = self.pred_lookup[pred_name]

            gt_manager = gt_person.get("manager")
            pred_manager = pred_person.get("manager")

            # Both have no manager (e.g., CEO)
            if gt_manager is None and pred_manager is None:
                correct += 1
                total += 1
                continue

            # One has manager, other doesn't
            if (gt_manager is None) != (pred_manager is None):
                total += 1
                errors.append(
                    {
                        "employee": gt_name,
                        "expected_manager": gt_manager,
                        "got_manager": pred_manager,
                        "type": "null_mismatch",
                    }
                )
                continue

            # Both have managers - check if they match through mapping
            total += 1
            if gt_manager in name_mapping:
                expected_pred_manager = name_mapping[gt_manager]
                if pred_manager == expected_pred_manager:
                    correct += 1
                else:
                    errors.append(
                        {
                            "employee": gt_name,
                            "expected_manager": gt_manager,
                            "got_manager": pred_manager,
                            "type": "wrong_manager",
                        }
                    )
            else:
                # Manager not in mapping - can't evaluate
                errors.append(
                    {
                        "employee": gt_name,
                        "expected_manager": gt_manager,
                        "got_manager": pred_manager,
                        "type": "manager_not_in_mapping",
                    }
                )

        accuracy = (correct / total * 100) if total > 0 else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors,
        }

    def evaluate(self) -> EvaluationResults:
        """Run full evaluation"""
        # 1. Name matching
        name_mapping, unmatched_gt, unmatched_pred = self.fuzzy_match_names()

        # 2. Coverage
        coverage = self.evaluate_coverage(name_mapping, unmatched_gt, unmatched_pred)

        # 3. Manager accuracy
        manager_eval = self.evaluate_managers(name_mapping)

        return EvaluationResults(
            name_mapping=name_mapping,
            unmatched_gt=unmatched_gt,
            unmatched_pred=unmatched_pred,
            coverage_pct=coverage["coverage_pct"],
            manager_accuracy=manager_eval["accuracy"],
            manager_correct=manager_eval["correct"],
            manager_total=manager_eval["total"],
            manager_errors=manager_eval["errors"],
        )


def _normalize_name(name: str) -> str:
    """Remove parentheses and their content, then strip whitespace."""
    return re.sub(r"\s*\([^)]*\)", "", name).strip()


def print_results(results: EvaluationResults):
    """Pretty print evaluation results to terminal"""

    # Header
    print("\n" + "=" * 80)
    print("ORG CHART EVALUATION RESULTS".center(80))
    print("=" * 80 + "\n")

    # 1. Name Matching Results
    print("📊 NAME MATCHING")
    print("-" * 80)
    total_gt = len(results.name_mapping) + len(results.unmatched_gt)
    total_pred = len(results.name_mapping) + len(results.unmatched_pred)
    print(f"  Ground Truth Names:  {total_gt}")
    print(f"  Predicted Names:     {total_pred}")
    print(
        f"  Matched:             {len(results.name_mapping)} ({results.coverage_pct:.1f}%)"
    )
    print(f"  Unmatched (GT):      {len(results.unmatched_gt)}")
    print(f"  Unmatched (Pred):     {len(results.unmatched_pred)}")

    if results.unmatched_gt:
        print(f"\n  Missing from Predicted ({len(results.unmatched_gt)}):")
        for name in sorted(results.unmatched_gt)[:10]:  # Show first 10
            print(f"    • {name}")
        if len(results.unmatched_gt) > 10:
            print(f"    ... and {len(results.unmatched_gt) - 10} more")

    if results.unmatched_pred:
        print(f"\n  Extra in Predicted ({len(results.unmatched_pred)}):")
        for name in sorted(results.unmatched_pred)[:10]:
            print(f"    • {name}")
        if len(results.unmatched_pred) > 10:
            print(f"    ... and {len(results.unmatched_pred) - 10} more")

    # 2. Coverage Metrics
    print("\n" + "=" * 80)
    print("📈 COVERAGE METRICS")
    print("-" * 80)
    print(
        f"  Employee Coverage: {results.coverage_pct:.1f}% ({len(results.name_mapping)}/{total_gt})"
    )

    # 3. Manager Relationship Accuracy
    print("\n" + "=" * 80)
    print("👔 MANAGER RELATIONSHIP ACCURACY")
    print("-" * 80)
    print(
        f"  Accuracy: {results.manager_accuracy:.1f}% ({results.manager_correct}/{results.manager_total})"
    )
    print(f"  Correct:  {results.manager_correct}")
    print(f"  Errors:   {len(results.manager_errors)}")

    if results.manager_errors:
        print(f"\n  Manager Errors ({len(results.manager_errors)}):")

        # Group by error type
        by_type = {}
        for error in results.manager_errors:
            error_type = error["type"]
            if error_type not in by_type:
                by_type[error_type] = []
            by_type[error_type].append(error)

        # Show errors by type
        for error_type, errors in sorted(by_type.items()):
            print(f"\n  {error_type.replace('_', ' ').title()} ({len(errors)}):")
            for error in errors[:5]:  # Show first 5 per type
                emp = error["employee"]
                expected = error["expected_manager"] or "None"
                got = error["got_manager"] or "None"
                print(f"    • {emp}")
                print(f"      Expected: {expected}")
                print(f"      Got:      {got}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")

    # Summary
    print("\n" + "=" * 80)
    print("✅ OVERALL SUMMARY")
    print("-" * 80)
    print(f"  Name Matching:       {results.coverage_pct:.1f}%")
    print(f"  Manager Accuracy:    {results.manager_accuracy:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated org chart against ground truth",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to the predicted/generated org chart markdown file",
    )
    parser.add_argument(
        "--true", required=True, help="Path to the ground truth org chart markdown file"
    )

    args = parser.parse_args()

    # Validate files exist
    predicted = OrgChart.from_md_file(args.pred)
    ground_truth = OrgChart.from_md_file(args.true)

    # Run evaluation
    evaluator = OrgChartEvaluator(predicted, ground_truth)
    results = evaluator.evaluate()

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
