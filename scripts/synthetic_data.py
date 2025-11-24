import random
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
N_INTERVIEWS = 5000
OUTPUT_FILE = "fake_firm_interviews.csv"

random.seed(42)

# -----------------------------
# FIXED SET OF 60 FIRMS
# -----------------------------
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

industries = [
    "technology", "semiconductors", "financial services", "consumer goods",
    "energy", "transportation and logistics", "healthcare", "retail",
    "automotive", "aerospace and defense", "telecommunications", "software",
    "industrial manufacturing"
]

topics = [
    "artificial intelligence", "automation", "supply chain resilience",
    "inflation and interest rates", "labor shortages", "remote work",
    "regulation and compliance", "climate risk and ESG", "digital transformation",
    "data privacy", "cybersecurity", "geopolitical tensions", "export controls",
    "tariffs", "consumer demand", "capital allocation", "R&D investments",
    "reshoring and nearshoring", "cloud infrastructure", "platform business models"
]

macro_forces = [
    "the pandemic", "rising interest rates", "geopolitical competition",
    "U.S.-China tensions", "regulatory uncertainty", "energy price volatility",
    "tight labor markets", "rapid digitization", "shifts in consumer behavior",
    "supply chain disruptions"
]


# -----------------------------
# DATE GENERATION
# -----------------------------
def random_date(start_year=2010, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    offset = random.randint(0, delta.days)
    return (start + timedelta(days=offset)).date().isoformat()


# -----------------------------
# INTERVIEW TEXT GENERATION
# -----------------------------
def make_title(company, industry):
    patterns = [
        f"{company} discusses strategy shifts in the {industry} sector",
        f"{company} on navigating {random.choice(macro_forces)}",
        f"{company}: 'This is a turning point for {industry}'",
        f"{company} outlines priorities in {random.choice(topics)}"
    ]
    return random.choice(patterns)

def make_question(company, industry, topic):
    templates = [
        f"Interviewer: How has {topic} changed the way {company} operates in the {industry} space?",
        f"Interviewer: What do stakeholders ask most about regarding {topic}?",
        f"Interviewer: Where does {topic} fit into your long-term strategy?",
        f"Interviewer: Has {topic} been more of a challenge or an opportunity for {company}?",
        f"Interviewer: How does leadership communicate internally about {topic}?"
    ]
    return random.choice(templates)

def make_answer(company, industry, topic):
    templates = [
        f"{company}: {topic} is no longer simply an initiative; it has become part of our operational foundation. "
        f"In the {industry} sector, adapting quickly is not optional—it’s essential.",

        f"{company}: We think of {topic} in terms of capability, risk, and long-term positioning. "
        f"It influences capital allocation, workforce training, and customer engagement.",

        f"{company}: One challenge with {topic} is separating real value from hype. "
        f"We focus on measurable operational gains rather than short-term excitement.",

        f"{company}: Many people assume {topic} is purely technological. In reality, it requires cultural alignment "
        f"and consistent communication across teams.",

        f"{company}: The last few years—especially {random.choice(macro_forces)}—made it clear that "
        f"resilience matters as much as efficiency. {topic} is central to that shift."
    ]
    return random.choice(templates)

def make_closing(company, industry):
    templates = [
        f"{company}: Our commitment is to stay disciplined and focus on long-term execution. "
        f"The {industry} landscape will continue to evolve, but our strategy remains grounded.",

        f"{company}: What gives us confidence is the strength of our teams and the opportunities ahead. "
        f"We believe in balancing ambition with operational precision.",

        f"{company}: This is a period where firms that combine innovation with stable leadership "
        f"will define the next era of competition in the {industry} sector."
    ]
    return random.choice(templates)


def generate_interview_text(company, industry, length_type):
    n_topics = random.randint(3, 5)
    chosen_topics = random.sample(topics, n_topics)

    # medium: 8–14 sentences, long: 15–24
    if length_type == "medium":
        target_sentences = random.randint(8, 14)
    else:
        target_sentences = random.randint(15, 24)

    sentences = []

    # Intro paragraph
    intro = (
        f"The conversation took place at {company}'s headquarters in the United States, "
        f"where leadership reflected on navigating a rapidly changing {industry} landscape. "
        f"Throughout the discussion, recurring themes included resilience, discipline, "
        f"and long-term positioning."
    )
    sentences.append(intro)

    # Q & A
    while len(sentences) < target_sentences:
        topic = random.choice(chosen_topics)
        q = make_question(company, industry, topic)
        a1 = make_answer(company, industry, topic)

        sentences.append(q)
        sentences.append(a1)

        if len(sentences) < target_sentences and random.random() < 0.6:
            follow = (
                f"Interviewer: You previously mentioned that transformation doesn't happen overnight. "
                f"How far along is {company} in this process?"
            )
            a2 = (
                f"{company}: We are far enough to see meaningful improvements, but early enough "
                f"that each quarter still brings new insights. Rewiring organizational habits takes time."
            )
            sentences.append(follow)
            sentences.append(a2)

        if len(sentences) >= target_sentences:
            break

    # Closing
    sentences.append(make_closing(company, industry))

    return " ".join(sentences)


# -----------------------------
# MAIN GENERATION LOOP
# -----------------------------
records = []

for i in range(1, N_INTERVIEWS + 1):
    industry = random.choice(industries)
    company = random.choice(firm_names)
    date = random_date()
    length_type = random.choice(["medium", "long"])

    title = make_title(company, industry)
    text = generate_interview_text(company, industry, length_type)

    interview_id = f"INT_{i:05d}"
    filename = f"interview_{i:05d}.xml"

    records.append({
        "interview_id": interview_id,
        "filename": filename,
        "title": title,
        "company": company,
        "industry": industry,
        "date": date,
        "length_type": length_type,
        "text": text
    })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"Generated {len(df)} synthetic firm interviews.")
print(f"Saved to: {OUTPUT_FILE}")
print(df.head(3))
