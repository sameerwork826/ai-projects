#!/usr/bin/env python3
"""
Knowledge Distillation Training Script

This script implements knowledge distillation for model compression on MNIST dataset.
It trains a teacher model and then uses it to train a smaller student model.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import os
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader


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


def load_data(batch_size=64):
    """Load and prepare MNIST dataset"""
    print("Loading MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    
    return trainloader, testloader


def evaluate(model, testloader):
    """Evaluate model accuracy on test set"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def knowledge_distillation_loss(student_logits, teacher_logits):
    """Calculate knowledge distillation loss using KL divergence"""
    p_teacher = F.softmax(teacher_logits, dim=1)
    p_student = F.log_softmax(student_logits, dim=1)
    loss = F.kl_div(p_student, p_teacher, reduction='batchmean')
    return loss


def train_teacher(model, trainloader, testloader, epochs=5, lr=0.001):
    """Train the teacher model"""
    print("\n" + "="*50)
    print("TRAINING TEACHER MODEL")
    print("="*50)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for data in progress_bar:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        accuracy = evaluate(model, testloader)
        avg_loss = running_loss / len(trainloader)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print()
    
    final_accuracy = evaluate(model, testloader)
    print(f"Final Teacher Accuracy: {final_accuracy * 100:.2f}%")
    return final_accuracy


def train_student(student_model, teacher_model, trainloader, testloader, epochs=5, lr=0.001):
    """Train the student model using knowledge distillation"""
    print("\n" + "="*50)
    print("TRAINING STUDENT MODEL (Knowledge Distillation)")
    print("="*50)
    
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    teacher_model.eval()  # Set teacher to evaluation mode
    
    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for data in progress_bar:
            inputs, labels = data
            optimizer.zero_grad()
            
            # Get student predictions
            student_logits = student_model(inputs)
            
            # Get teacher predictions (detached to avoid backprop)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            # Calculate knowledge distillation loss
            loss = knowledge_distillation_loss(student_logits, teacher_logits)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        accuracy = evaluate(student_model, testloader)
        avg_loss = running_loss / len(trainloader)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print()
    
    final_accuracy = evaluate(student_model, testloader)
    print(f"Final Student Accuracy: {final_accuracy * 100:.2f}%")
    return final_accuracy


def benchmark_models(teacher_model, student_model, testloader):
    """Benchmark inference speed of both models"""
    print("\n" + "="*50)
    print("BENCHMARKING INFERENCE SPEED")
    print("="*50)
    
    # Warm up
    dummy_input = torch.randn(1, 1, 28, 28)
    for _ in range(10):
        _ = teacher_model(dummy_input)
        _ = student_model(dummy_input)
    
    # Benchmark teacher
    start_time = time()
    teacher_model.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, _ = data
            _ = teacher_model(inputs)
    teacher_time = time() - start_time
    
    # Benchmark student
    start_time = time()
    student_model.eval()
    with torch.no_grad():
        for data in testloader:
            inputs, _ = data
            _ = student_model(inputs)
    student_time = time() - start_time
    
    print(f"Teacher Model Inference Time: {teacher_time:.2f}s")
    print(f"Student Model Inference Time: {student_time:.2f}s")
    print(f"Speed Improvement: {teacher_time/student_time:.2f}x")
    
    return teacher_time, student_time


def save_models(teacher_model, student_model):
    """Save trained models"""
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models
    torch.save(teacher_model.state_dict(), 'models/teacher_model.pth')
    torch.save(student_model.state_dict(), 'models/student_model.pth')
    
    print("Models saved to:")
    print("  - models/teacher_model.pth")
    print("  - models/student_model.pth")


def main():
    """Main training pipeline"""
    print("Knowledge Distillation Training Pipeline")
    print("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    trainloader, testloader = load_data()
    
    # Initialize models
    teacher_model = TeacherNet().to(device)
    student_model = StudentNet().to(device)
    
    # Move data to device
    def to_device(data):
        return data[0].to(device), data[1].to(device)
    
    # Train teacher model
    teacher_accuracy = train_teacher(teacher_model, trainloader, testloader)
    
    # Train student model with knowledge distillation
    student_accuracy = train_student(student_model, teacher_model, trainloader, testloader)
    
    # Benchmark models
    teacher_time, student_time = benchmark_models(teacher_model, student_model, testloader)
    
    # Save models
    save_models(teacher_model, student_model)
    
    # Final summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Teacher Model Accuracy: {teacher_accuracy * 100:.2f}%")
    print(f"Student Model Accuracy: {student_accuracy * 100:.2f}%")
    print(f"Accuracy Retention: {(student_accuracy/teacher_accuracy) * 100:.2f}%")
    print(f"Teacher Inference Time: {teacher_time:.2f}s")
    print(f"Student Inference Time: {student_time:.2f}s")
    print(f"Speed Improvement: {teacher_time/student_time:.2f}x")
    
    # Count parameters
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"Teacher Parameters: {teacher_params:,}")
    print(f"Student Parameters: {student_params:,}")
    print(f"Parameter Reduction: {(1 - student_params/teacher_params) * 100:.2f}%")


if __name__ == "__main__":
    main()
