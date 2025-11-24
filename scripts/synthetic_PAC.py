import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

# ------------------------------------------------
# FIRMS (same 60 as interviews dataset)
# ------------------------------------------------
firm_names = [
    "Apex Systems", "Pioneer Energy", "Summit Technologies", "Liberty Capital",
    "Harbor Retail Group", "Frontier Semiconductor", "Sterling Logistics",
    "Atlas Health", "Aurora Networks", "Crescent Automotive", "Northbridge Holdings",
    "Oakstone Industries", "BlueRiver Manufacturing", "Ironwood Aerospace",
    "SilverRock Solutions", "Redwood Partners", "Skyline Retail", "Longview Energy",
    "Riverstone Technologies", "Brightline Capital", "Greenfield Manufacturing",
    "Heritage Automotive", "Cypress Health", "Evercrest Systems",
    "Mountain Ridge Industries", "GoldenGate Networks", "Sapphire Energy",
    "Falcon Semiconductor", "Highland Logistics", "Mosaic Consumer Goods",
    "EaglePeak Aerospace", "MetroTech Solutions", "UrbanStar Retail",
    "AmeriCore Energy", "NovaBridge Partners", "Quantum Systems",
    "Vector Automotive", "PrimeHealth Corporation", "United Semiconductor",
    "Centurion Capital", "Zenith Industries", "TerraLine Logistics",
    "NorthStar Retail Group", "Titan Aerospace", "Vanguard Networks",
    "Equinox Manufacturing", "BluePeak Technologies", "Omega Health",
    "Aspire Automotive", "Horizon Capital", "NextWave Semiconductor",
    "Paramount Systems", "SummitLine Logistics", "Vertex Energy",
    "Cobalt Holdings", "Everton Technologies", "Millennium Retail Group",
    "Infinity Partners", "Arclight Manufacturing", "Orbit Semiconductor"
]

# ------------------------------------------------
# Synthetic “latent ideology” for each firm
# (YOU WILL USE THIS AS YOUR SUPERVISION SIGNAL)
# ------------------------------------------------
firm_ideology = {}

for f in firm_names:
    # moderate skew: tech firms lean left, energy & defense lean right
    if "Energy" in f or "Aerospace" in f or "Semiconductor" in f:
        lean = random.uniform(0.1, 0.8)    # right-leaning
    elif "Health" in f or "Retail" in f or "Consumer" in f:
        lean = random.uniform(-0.4, 0.4)   # centrist
    else:
        lean = random.uniform(-0.8, -0.1)  # left-leaning
    firm_ideology[f] = lean  # continuous [-1, +1]


# ------------------------------------------------
# Helper functions
# ------------------------------------------------
def random_date(start=2010, end=2024):
    start_dt = datetime(start, 1, 1)
    end_dt = datetime(end, 12, 31)
    delta = end_dt - start_dt
    return (start_dt + timedelta(days=random.randint(0, delta.days))).date()

def draw_amount():
    """Right-skewed PAC contribution distribution."""
    base = random.lognormvariate(7, 0.7)       # lognormal
    return round(min(base, 25000), 2)          # cap at $25k (FEC-like)


def draw_party(ideology_score):
    """More Republican contributions if score > 0, more Democratic if < 0."""
    p_rep = (ideology_score + 1) / 2           # map [-1,+1] → [0,1]
    return "R" if random.random() < p_rep else "D"


# ------------------------------------------------
# Generate the synthetic PAC dataset
# ------------------------------------------------
N_ROWS = 30000  # You can increase to 50k or 100k if needed

records = []

for _ in range(N_ROWS):
    firm = random.choice(firm_names)
    ideology = firm_ideology[firm]
    party = draw_party(ideology)
    amount = draw_amount()

    records.append({
        "firm": firm,
        "date": random_date(),
        "amount": amount,
        "party": party,
        "ideology_score": ideology  # continuous label for training
    })

df = pd.DataFrame(records)
df.to_csv("synthetic_pac_contributions.csv", index=False)

print("Generated:", len(df), "rows")
print(df.head())
