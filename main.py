import argparse
from src.train_feature_model import train_feature_model
from src.evaluate_feature_model import (
    evaluate_feature_model,
    predict_single_image
)
from src.visualize_patch_score import visualize_image_score

def main():
    parser = argparse.ArgumentParser(
        description="Industrial Defect Detection - Feature Based Anomaly Detection"
    )

    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=["bottle", "metal_nut", "tile"],
        help="Category to train/evaluate"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["feature_train", "feature_evaluate", "feature_predict", "feature_visualize"],
        help="Operation mode"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image (required for feature_predict mode)"
    )

    args = parser.parse_args()

    if args.mode == "feature_train":
        print(f"\nTraining feature model for category: {args.category}")
        train_feature_model(args.category)

    elif args.mode == "feature_evaluate":
        print(f"\nEvaluating feature model for category: {args.category}")
        evaluate_feature_model(args.category)

    elif args.mode == "feature_predict":
        if args.image is None:
            print("Error: Please provide --image path for prediction mode.")
        else:
            print(f"\nPredicting image: {args.image}")
            predict_single_image(args.category, args.image)

    elif args.mode == "feature_visualize":
        if args.image is None:
            print("Error: Please provide --image path for visualization mode.")
        else:
            print(f"\nVisualizing image: {args.image}")
            visualize_image_score(args.category, args.image)


if __name__ == "__main__":
    main()