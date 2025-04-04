import argparse
from datetime import date
import logging

from duration_prediction.train import train

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train a model based on specified dates and save it to a given path")
    parser.add_argument("--train-date", required=True, help="Training month in YYYY-MM format")
    parser.add_argument("--val-date", required=True, help="Validation month in YYYY-MM format")
    parser.add_argument("--model-save-path", required=True, help="Path where the trained model is saved")
    
    args = parser.parse_args()
    
    train_year, train_month = args.train_date.split("-")
    train_year = int(train_year)
    train_month = int(train_month)

    val_year, val_month = args.val_date.split("-")
    val_year = int(val_year)
    val_month = int(val_month)


    train_date = date(train_year, train_month, 1)
    val_date = date(val_year, val_month, 1)
    out_path = args.model_save_path
    
    logger.info(f"Running training: train: {train_date}, val: {val_date}, saving to: {out_path}")
    train(train_date=train_date, val_date=val_date, out_path=out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
