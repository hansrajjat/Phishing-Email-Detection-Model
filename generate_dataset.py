import pandas as pd
import random
import os

def generate_mock_dataset(num_samples=1000):
    """
    Generates a mock dataset of emails for training the Phishing Email Detection System.
    """
    safe_emails = [
        "Hi there, just checking in to see if we are still on for tomorrow's meeting.",
        "Please find attached the report for Q3. Let me know if you need any changes.",
        "Your order #12345 has been shipped. Track your package here: https://store.example.com/track",
        "Happy birthday! Hope you have a wonderful day.",
        "Can we reschedule our 1 PM call to 3 PM? Thanks.",
        "The project deadline has been extended to next Friday.",
        "Here are the meeting notes from yesterday's discussion.",
        "Don't forget to submit your timesheet by the end of the day.",
        "Are you available for a quick chat later this afternoon?",
        "Thank you for your purchase. Your receipt is attached."
    ]

    phishing_emails = [
        "URGENT: Your account has been compromised. Please verify your password immediately at http://192.168.1.1/login",
        "Security Alert: We detected suspicious login attempts. Click here to update your account: http://secure-update-now.com",
        "You have won a $1000 gift card! Click the link to claim your prize before it expires.",
        "Important notification regarding your bank account. Please login to verify your identity: http://www.bank-verify-security.com",
        "Your PayPal account is limited. Update now to restore access. http://paypal-update.example.net",
        "Action Required: Your email quota is almost full. Click here to upgrade your storage.",
        "We noticed unusual activity on your credit card. Verify recent transactions here: https://fraud-alert-check.com",
        "Final Warning: Your subscription will be canceled unless you update your payment information today.",
        "Congratulations! You are the lucky winner of our annual giveaway. Claim your reward now.",
        "Dear customer, your invoice is overdue. Please pay immediately using the secure link provided."
    ]

    data = []
    
    # Generate balanced dataset
    for _ in range(num_samples // 2):
        data.append({"text": random.choice(safe_emails), "label": "Safe"})
        data.append({"text": random.choice(phishing_emails), "label": "Phishing"})
        
    df = pd.DataFrame(data)
    
    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/emails.csv', index=False)
    print(f"Successfully generated {num_samples} mock emails in dataset/emails.csv")

if __name__ == "__main__":
    generate_mock_dataset(1000)
