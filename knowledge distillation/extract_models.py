#!/usr/bin/env python3
"""
Model Extraction Script

This script extracts the trained models from the notebook and saves them for use in the Streamlit app.
Run this after training the models in the notebook.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class TeacherNet(nn.Module):
    """Teacher model with convolutional layers"""
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(5, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StudentNet(nn.Module):
    """Student model with only fully connected layers"""
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def extract_models_from_notebook():
    """
    Extract models from notebook variables.
    This function should be run in the same environment as the notebook.
    """
    print("Extracting models from notebook...")
    
    # Check if models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created models directory")
    
    try:
        # Try to access the models from the notebook environment
        # These variables should be available if the notebook was run
        if 'teacher_model' in globals():
            torch.save(teacher_model.state_dict(), 'models/teacher_model.pth')
            print("✅ Teacher model saved to models/teacher_model.pth")
        else:
            print("❌ Teacher model not found. Please run the notebook first.")
            
        if 'student_model' in globals():
            torch.save(student_model.state_dict(), 'models/student_model.pth')
            print("✅ Student model saved to models/student_model.pth")
        else:
            print("❌ Student model not found. Please run the notebook first.")
            
    except NameError as e:
        print(f"❌ Error: {e}")
        print("Please run this script in the same environment as the notebook, or run train_models.py instead.")


def create_dummy_models():
    """
    Create dummy models for demonstration purposes.
    Use this if you want to test the app without training.
    """
    print("Creating dummy models for demonstration...")
    
    # Check if models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created models directory")
    
    # Create dummy models
    teacher_model = TeacherNet()
    student_model = StudentNet()
    
    # Save dummy models
    torch.save(teacher_model.state_dict(), 'models/teacher_model.pth')
    torch.save(student_model.state_dict(), 'models/student_model.pth')
    
    print("✅ Dummy models created and saved")
    print("Note: These are untrained models for demonstration only.")
    print("For real performance, please run train_models.py or train in the notebook.")


if __name__ == "__main__":
    print("Model Extraction Script")
    print("=" * 30)
    
    choice = input("Choose an option:\n1. Extract from notebook (requires notebook to be run)\n2. Create dummy models\nEnter choice (1 or 2): ")
    
    if choice == "1":
        extract_models_from_notebook()
    elif choice == "2":
        create_dummy_models()
    else:
        print("Invalid choice. Creating dummy models...")
        create_dummy_models()
