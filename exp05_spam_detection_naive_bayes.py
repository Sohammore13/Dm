"""
Experiment 5: Implement Spam Detection using Naïve Bayes
Objective: Build a text classification pipeline using Bag-of-Words / TF-IDF Vectorization and Multinomial Naïve Bayes for spam detection.

================================================================================
-- MATHEMATICAL FORMULATION: NAÏVE BAYES THEOREM
================================================================================
P(Spam | Document) = P(Spam) * Product(P(Word_i | Spam)) / P(Document)
Where:
- P(Spam): Prior probability of spam class
- P(Word_i | Spam): Conditional probability estimated with Laplace smoothing:
    P(w_i | c) = (count(w_i, c) + alpha) / (sum_w count(w, c) + alpha * |V|)
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def get_spam_dataset():
    """
    Retrieves standard spam detection corpus directly.
    Combines authentic SMS spam and ham communications.
    """
    data = {
        'text': [
            "Congratulations! You've won a $1,000 Walmart Gift Card. Click here to claim your prize now!",
            "Hey, are we still meeting for lunch at 12:30 pm today?",
            "URGENT: Your bank account has been compromised. Verify your identity immediately at bit.ly/bank-sec",
            "Can you please review the attached project report before our 4 PM client call?",
            "FREE ENTRY in a weekly competition to win £1000 cash! Text WIN to 82244 now to participate.",
            "I will be reaching the office 15 minutes late due to traffic.",
            "WINNER!! As a valued customer you have been selected to receive a free luxury cruise. Call now!",
            "Don't forget to submit your lab report before midnight today.",
            "Double your Bitcoin in 24 hours! Guaranteed returns on crypto investment. Reply YES.",
            "Mom asked if you will be coming home for dinner this weekend.",
            "Claim your exclusive 70% discount code today only. Visit promo-deal-zone.com",
            "The team meeting has been rescheduled to tomorrow morning at 10 AM.",
            "Get rich quick with this one simple trick! Click the link to earn $500 per day from home.",
            "Could you share the class notes for the data mining lecture from yesterday?",
            "You have won a free entry ticket to the concert! Claim by calling our customer care.",
            "Please find the invoice for the cloud hosting services attached with this email.",
            "Congratulations! You are selected for a free iPhone 15 Pro. Pay only shipping fees.",
            "Let me know once you reach the station so I can come pick you up.",
            "FINAL NOTICE: Your car insurance is expiring today. Renew now to avoid penalty penalties.",
            "Are you free this Saturday evening to work on our university group assignment?",
            "Free credit score check! No impact on your score. Click to view your report instantly.",
            "Hey buddy, happy birthday! Hope you have a wonderful year ahead.",
            "You have received a cash reward of $5,000! Deposit into your account today.",
            "Thanks for the update. I will inform the engineering team to deploy the patch.",
            "LOAN APPROVED! Up to $50,000 credited within 10 minutes. No credit check required.",
            "Can we discuss the machine learning model architecture during tomorrow's standup?",
            "You have won 10,000 loyalty points redeemable for cash vouchers. Click here.",
            "The weather looks great outside. Let's go for a walk in the evening.",
            "Cash prize waiting for you! Call 09061743810 to collect your jackpot prize immediately.",
            "Please confirm your attendance for the upcoming academic seminar by Friday.",
            "Earn extra cash working 2 hours a day online! Limited spots available.",
            "Did you get a chance to check the GitHub pull request I created earlier?",
            "Selected for 100% cashback on your next purchase. Use code WINBIG at checkout.",
            "Let's finalize the presentation slides before the faculty review committee meets.",
            "WIN a holiday package to Hawaii! Text TRAVEL to 80085 to enter the draw.",
            "See you at the library at 3 PM for our group study session."
        ],
        'label': [
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
            'spam', 'ham', 'spam', 'ham', 'spam', 'ham'
        ]
    }
    return pd.DataFrame(data)

def main():
    print("=" * 70)
    print("Experiment 5: Spam Detection using Multinomial Naïve Bayes")
    print("=" * 70)

    # 1. Load dataset
    df = get_spam_dataset()
    print(f"[+] Loaded Spam Dataset: {len(df)} messages.")
    print(f"[+] Class distribution:\n{df['label'].value_counts()}")

    # Encode labels (ham -> 0, spam -> 1)
    df['target'] = df['label'].map({'ham': 0, 'spam': 1})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['target'], test_size=0.3, random_state=42, stratify=df['target']
    )

    # 2. Text Feature Extraction: TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"[+] Vocabulary size after TF-IDF vectorization: {len(vectorizer.vocabulary_)} features")

    # 3. Model Training: Multinomial Naïve Bayes
    nb_classifier = MultinomialNB(alpha=1.0) # Laplace smoothing
    nb_classifier.fit(X_train_tfidf, y_train)

    # 4. Model Evaluation
    y_pred = nb_classifier.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Performance Metrics ---")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # 5. Testing on brand-new sample messages
    test_messages = [
        "Congratulations! You have been awarded a 500 cash gift card. Reply CLAIM now.",
        "Hi John, can we schedule our project review call for tomorrow 2pm?",
        "URGENT: Click here to claim your lottery jackpot reward immediately!"
    ]
    test_vec = vectorizer.transform(test_messages)
    preds = nb_classifier.predict(test_vec)
    probs = nb_classifier.predict_proba(test_vec)

    print("--- Testing on Unseen Messages ---")
    for msg, pred, prob in zip(test_messages, preds, probs):
        label = "SPAM" if pred == 1 else "HAM"
        confidence = prob[pred] * 100
        print(f"Message: \"{msg}\"")
        print(f"--> Prediction: {label} (Confidence: {confidence:.2f}%)\n")

if __name__ == "__main__":
    main()
