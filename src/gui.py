import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_email

class PhishingDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phishing Email Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg="#f4f4f9")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title Label
        title_label = tk.Label(self.root, text="Phishing Email Detection System", font=("Helvetica", 20, "bold"), bg="#f4f4f9", fg="#333333")
        title_label.pack(pady=20)
        
        # Input Section
        input_frame = tk.Frame(self.root, bg="#f4f4f9")
        input_frame.pack(padx=40, fill=tk.BOTH, expand=True)
        
        instruction_label = tk.Label(input_frame, text="Paste the email content below:", font=("Helvetica", 12), bg="#f4f4f9", fg="#555555")
        instruction_label.pack(anchor="w", pady=(0, 5))
        
        self.text_box = tk.Text(input_frame, height=12, font=("Consolas", 11), bd=1, relief="solid", wrap="word")
        self.text_box.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Buttons Section
        btn_frame = tk.Frame(self.root, bg="#f4f4f9")
        btn_frame.pack(pady=10)
        
        analyze_btn = tk.Button(btn_frame, text="Analyze Email", font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", width=15, command=self.analyze_email, relief="flat", cursor="hand2")
        analyze_btn.grid(row=0, column=0, padx=10)
        
        clear_btn = tk.Button(btn_frame, text="Clear", font=("Helvetica", 12), bg="#f44336", fg="white", width=10, command=self.clear_text, relief="flat", cursor="hand2")
        clear_btn.grid(row=0, column=1, padx=10)
        
        load_sample_btn = tk.Button(btn_frame, text="Load Sample Email", font=("Helvetica", 12), bg="#2196F3", fg="white", width=20, command=self.load_sample, relief="flat", cursor="hand2")
        load_sample_btn.grid(row=0, column=2, padx=10)
        
        # Output Section
        output_frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief="solid")
        output_frame.pack(padx=40, pady=20, fill=tk.X)
        
        self.prediction_label = tk.Label(output_frame, text="Prediction: -", font=("Helvetica", 16, "bold"), bg="#ffffff")
        self.prediction_label.pack(pady=(15, 5))
        
        self.confidence_label = tk.Label(output_frame, text="Confidence: -", font=("Helvetica", 14), bg="#ffffff")
        self.confidence_label.pack(pady=5)
        
        self.risk_label = tk.Label(output_frame, text="Risk Level: -", font=("Helvetica", 14), bg="#ffffff")
        self.risk_label.pack(pady=(5, 15))
        
    def analyze_email(self):
        email_content = self.text_box.get("1.0", tk.END).strip()
        
        if not email_content:
            messagebox.showwarning("Empty Input", "Please enter some email text to analyze.")
            return
            
        status, confidence_pct, prob_phishing_pct = predict_email(email_content)
        
        if status == "ERROR":
            messagebox.showerror("Error", "Could not load the model. Please ensure you have trained the model first.")
            return
            
        # Update UI
        self.prediction_label.config(text=f"Prediction: {status}")
        self.confidence_label.config(text=f"Confidence: {confidence_pct:.1f}%")
        
        # Determine Risk Level and Colors
        if status == "PHISHING":
            self.prediction_label.config(fg="#d32f2f") # Red
            
            if prob_phishing_pct > 80:
                risk = "High Risk"
                risk_color = "#d32f2f"
            else:
                risk = "Medium Risk"
                risk_color = "#f57c00" # Orange
        else:
            self.prediction_label.config(fg="#388e3c") # Green
            
            if prob_phishing_pct < 20:
                risk = "Low Risk"
                risk_color = "#388e3c"
            else:
                risk = "Medium Risk"
                risk_color = "#f57c00"
                
        self.risk_label.config(text=f"Risk Level: {risk}", fg=risk_color)

    def clear_text(self):
        self.text_box.delete("1.0", tk.END)
        self.prediction_label.config(text="Prediction: -", fg="#333333")
        self.confidence_label.config(text="Confidence: -", fg="#333333")
        self.risk_label.config(text="Risk Level: -", fg="#333333")
        
    def load_sample(self):
        sample = """URGENT: Your account has been compromised!
Please click the link below to verify your identity and restore access immediately.
Failure to do so will result in permanent suspension of your account.
http://secure-update-now.com/login"""
        self.clear_text()
        self.text_box.insert(tk.END, sample)

def launch_gui():
    root = tk.Tk()
    app = PhishingDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    launch_gui()
