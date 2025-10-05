#!/usr/bin/env python3
"""
Knowledge Distillation Streamlit App

Interactive web application to showcase knowledge distillation models.
Allows users to compare teacher and student model performance and test with custom inputs.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os


# Model definitions (same as in training script)
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


@st.cache_resource
def load_models():
    """Load pre-trained models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models
    teacher_model = TeacherNet()
    student_model = StudentNet()
    
    # Load weights if available
    teacher_path = 'models/teacher_model.pth'
    student_path = 'models/student_model.pth'
    
    if os.path.exists(teacher_path) and os.path.exists(student_path):
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        student_model.load_state_dict(torch.load(student_path, map_location=device))
        st.success("✅ Pre-trained models loaded successfully!")
    else:
        st.warning("⚠️ Pre-trained models not found. Please run `python train_models.py` first.")
        st.info("Using randomly initialized models for demonstration.")
    
    teacher_model.eval()
    student_model.eval()
    
    return teacher_model, student_model, device


def preprocess_image(image):
    """Preprocess uploaded image for model input"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor


def predict_digit(model, image_tensor, device):
    """Get model prediction for an image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()


def create_drawing_canvas():
    """Create a drawing canvas for digit input"""
    st.subheader("🎨 Draw a Digit")
    
    # Create canvas using streamlit-drawable-canvas if available, otherwise use file upload
    try:
        from streamlit_drawable_canvas import st_canvas
        
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            # Convert canvas result to PIL Image
            image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            return image
    except ImportError:
        st.info("For drawing functionality, install streamlit-drawable-canvas: `pip install streamlit-drawable-canvas`")
        return None


def plot_model_comparison(teacher_probs, student_probs, digit):
    """Create comparison plot of model predictions"""
    digits = list(range(10))
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Teacher Model Predictions', 'Student Model Predictions'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Teacher predictions
    fig.add_trace(
        go.Bar(x=digits, y=teacher_probs, name='Teacher', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Student predictions
    fig.add_trace(
        go.Bar(x=digits, y=student_probs, name='Student', marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Model Predictions for Digit: {digit}",
        showlegend=False,
        height=400
    )
    
    fig.update_xaxes(title_text="Digit", row=1, col=1)
    fig.update_xaxes(title_text="Digit", row=1, col=2)
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=2)
    
    return fig


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Knowledge Distillation Showcase",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🧠 Knowledge Distillation Showcase")
    st.markdown("""
    This application demonstrates knowledge distillation, a model compression technique where a smaller 
    "student" model learns from a larger "teacher" model. Compare the performance of both models on 
    handwritten digit recognition.
    """)
    
    # Load models
    teacher_model, student_model, device = load_models()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Model information
    st.sidebar.subheader("📊 Model Information")
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    
    st.sidebar.metric("Teacher Parameters", f"{teacher_params:,}")
    st.sidebar.metric("Student Parameters", f"{student_params:,}")
    st.sidebar.metric("Parameter Reduction", f"{(1 - student_params/teacher_params) * 100:.1f}%")
    
    # Input method selection
    st.sidebar.subheader("📝 Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Image", "Draw Digit", "Sample Images"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Input")
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a handwritten digit image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a 28x28 grayscale image of a handwritten digit (0-9)"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=200)
                
        elif input_method == "Draw Digit":
            image = create_drawing_canvas()
            if image is not None:
                st.image(image, caption="Drawn Digit", width=200)
                
        elif input_method == "Sample Images":
            st.info("Sample images from MNIST test set will be used for demonstration.")
            # For demo purposes, we'll use a random sample
            sample_digit = st.selectbox("Select a digit to test:", range(10))
            # In a real implementation, you'd load actual MNIST samples
            image = None  # Placeholder
    
    with col2:
        st.subheader("🔍 Predictions")
        
        if (input_method in ["Upload Image", "Draw Digit"] and 'image' in locals() and image is not None) or input_method == "Sample Images":
            
            if input_method == "Sample Images":
                # Generate a random sample for demonstration
                st.info("Using random sample for demonstration")
                # Create a random 28x28 image
                random_image = np.random.rand(28, 28) * 255
                image = Image.fromarray(random_image.astype('uint8'))
                st.image(image, caption="Sample Image", width=200)
            
            # Preprocess image
            image_tensor = preprocess_image(image)
            
            # Get predictions
            with st.spinner("Getting predictions..."):
                start_time = time.time()
                
                teacher_pred, teacher_conf, teacher_probs = predict_digit(teacher_model, image_tensor, device)
                teacher_time = time.time() - start_time
                
                start_time = time.time()
                student_pred, student_conf, student_probs = predict_digit(student_model, image_tensor, device)
                student_time = time.time() - start_time
            
            # Display results
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.metric(
                    "Teacher Prediction",
                    f"{teacher_pred}",
                    f"Confidence: {teacher_conf:.3f}"
                )
                st.caption(f"Inference time: {teacher_time*1000:.1f}ms")
            
            with col_pred2:
                st.metric(
                    "Student Prediction", 
                    f"{student_pred}",
                    f"Confidence: {student_conf:.3f}"
                )
                st.caption(f"Inference time: {student_time*1000:.1f}ms")
            
            # Speed comparison
            if teacher_time > 0:
                speed_improvement = teacher_time / student_time
                st.metric("Speed Improvement", f"{speed_improvement:.2f}x")
            
            # Prediction comparison plot
            st.plotly_chart(
                plot_model_comparison(teacher_probs, student_probs, teacher_pred),
                use_container_width=True
            )
    
    # Model Architecture Comparison
    st.subheader("🏗️ Model Architecture Comparison")
    
    col_arch1, col_arch2 = st.columns(2)
    
    with col_arch1:
        st.markdown("""
        **Teacher Model (Complex)**
        ```
        Input (28x28) 
        ↓
        Conv2d(1, 32, 5) + ReLU
        ↓
        MaxPool2d(5, 5)
        ↓
        Flatten
        ↓
        Linear(512, 128) + ReLU
        ↓
        Linear(128, 10)
        ↓
        Output (10 classes)
        ```
        """)
    
    with col_arch2:
        st.markdown("""
        **Student Model (Simple)**
        ```
        Input (28x28)
        ↓
        Flatten
        ↓
        Linear(784, 128) + ReLU
        ↓
        Linear(128, 10)
        ↓
        Output (10 classes)
        ```
        """)
    
    # Knowledge Distillation Process
    st.subheader("🎓 Knowledge Distillation Process")
    
    st.markdown("""
    Knowledge distillation works by:
    
    1. **Teacher Training**: Train a complex teacher model using standard cross-entropy loss
    2. **Student Training**: Train a simpler student model using the teacher's "soft" predictions
    3. **Loss Function**: Use KL divergence to match the student's output distribution to the teacher's
    4. **Benefits**: Student model learns the teacher's knowledge while being smaller and faster
    
    **Key Formula:**
    ```
    Loss = KL_Divergence(Student_Softmax, Teacher_Softmax)
    ```
    """)
    
    # Performance Metrics (placeholder - would be loaded from actual results)
    st.subheader("📈 Performance Metrics")
    
    metrics_data = {
        'Model': ['Teacher', 'Student'],
        'Accuracy (%)': [98.81, 95.2],  # Placeholder values
        'Inference Time (ms)': [2.32, 1.88],  # Placeholder values
        'Parameters': [teacher_params, student_params]
    }
    
    col_metrics1, col_metrics2 = st.columns(2)
    
    with col_metrics1:
        st.dataframe(metrics_data, use_container_width=True)
    
    with col_metrics2:
        # Create a performance comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Accuracy (%)',
            x=metrics_data['Model'],
            y=metrics_data['Accuracy (%)'],
            yaxis='y',
            offsetgroup=1,
        ))
        
        fig.add_trace(go.Bar(
            name='Inference Time (ms)',
            x=metrics_data['Model'],
            y=metrics_data['Inference Time (ms)'],
            yaxis='y2',
            offsetgroup=2,
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis=dict(title='Accuracy (%)', side='left'),
            yaxis2=dict(title='Inference Time (ms)', side='right', overlaying='y'),
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About Knowledge Distillation:**
    
    Knowledge distillation is a model compression technique introduced by Hinton et al. (2015) that allows 
    a smaller "student" model to learn from a larger "teacher" model. This technique is particularly useful 
    for deploying models on resource-constrained devices while maintaining competitive performance.
    
    **References:**
    - Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network.
    - [Machine Learning Model Compression: A Critical Step Towards Efficient Deep Learning](https://www.dailydoseofds.com/model-compression-a-critical-step-towards-efficient-machine-learning)
    """)


if __name__ == "__main__":
    main()
