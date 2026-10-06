import numpy as np
import pandas as pd

def generate_transaction_data(n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic transaction telemetry dataset with statistical overlap.
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * 0.03)
    n_legit = n_samples - n_fraud
    
    # Legitimate transactions (includes some power-users and travel outliers)
    legit_amount = np.random.exponential(scale=65, size=n_legit) + 10
    legit_velocity = np.random.poisson(lam=1.8, size=n_legit)
    legit_ratio = np.random.normal(loc=1.0, scale=0.6, size=n_legit).clip(min=0.1)
    legit_foreign = np.random.binomial(n=1, p=0.08, size=n_legit)     # 8% legitimate travelers
    legit_new_device = np.random.binomial(n=1, p=0.12, size=n_legit)  # 12% legitimate new phones/laptops
    legit_night = np.random.binomial(n=1, p=0.08, size=n_legit)       # 8% late-night shoppers
    
    # Fraudulent transactions (includes stealth fraud trying to blend in)
    fraud_amount = np.random.exponential(scale=280, size=n_fraud) + 40
    fraud_velocity = np.random.poisson(lam=4.8, size=n_fraud) + 1
    fraud_ratio = np.random.normal(loc=3.2, scale=1.8, size=n_fraud).clip(min=0.5)
    fraud_foreign = np.random.binomial(n=1, p=0.55, size=n_fraud)
    fraud_new_device = np.random.binomial(n=1, p=0.60, size=n_fraud)
    fraud_night = np.random.binomial(n=1, p=0.32, size=n_fraud)
    
    df_legit = pd.DataFrame({
        'amount': legit_amount,
        'tx_velocity_1h': legit_velocity,
        'avg_amount_ratio': legit_ratio,
        'is_foreign_ip': legit_foreign,
        'is_new_device': legit_new_device,
        'is_night_transaction': legit_night,
        'is_fraud': 0
    })
    
    df_fraud = pd.DataFrame({
        'amount': fraud_amount,
        'tx_velocity_1h': fraud_velocity,
        'avg_amount_ratio': fraud_ratio,
        'is_foreign_ip': fraud_foreign,
        'is_new_device': fraud_new_device,
        'is_night_transaction': fraud_night,
        'is_fraud': 1
    })
    
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

if __name__ == "__main__":
    df = generate_transaction_data()
    print(f"Generated {len(df)} transactions with realistic statistical noise.")