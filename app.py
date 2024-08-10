import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go

# Initialize the text classification pipeline with the specified model
try:
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to create an interactive plot
def plot_emotion_scores(labels, scores, sentence):
    fig = go.Figure()

    # Define a color palette
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#F0F0F0']

    # Add bar trace
    fig.add_trace(go.Bar(
        x=labels,
        y=scores,
        text=[f'{score:.4f}' for score in scores],
        textposition='outside',
        marker=dict(color=colors[:len(labels)]),  # Use the color palette
        name='Emotion Scores'
    ))

    # Update layout
    fig.update_layout(
        title=f'Emotion Scores for Sentence: "{sentence}"',
        xaxis_title='Emotions',
        yaxis_title='Scores',
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, 1]),  # Assuming scores are between 0 and 1
        template='plotly_white'
    )
    
    fig.update_traces(marker=dict(line=dict(color='black', width=1.5)))

    return fig

# Streamlit interface
st.title('Emotion Classification and Visualization')

# Text input box
input_text = st.text_area("Enter a sentence for emotion classification:", "")

# History to store past inputs and results
if 'history' not in st.session_state:
    st.session_state.history = []

if st.button("Classify Emotion"):
    if input_text:
        # Get model outputs
        try:
            model_outputs = classifier([input_text])
        except Exception as e:
            st.error(f"Error during classification: {e}")
            st.stop()
        
        # Extract labels and scores
        output = model_outputs[0]
        labels = [emotion['label'] for emotion in output]
        scores = [emotion['score'] for emotion in output]
        
        # Print emotions and scores
        st.write(f"Sentence: '{input_text}'")
        st.write("Predictions:")
        for emotion in output:
            st.write(f"  {emotion['label']}: {emotion['score']:.4f}")

        # Plotting the emotions for the current sentence
        fig = plot_emotion_scores(labels, scores, input_text)
        st.plotly_chart(fig)

        # Aggregate and print emotion scores
        total_score = sum(scores)
        most_prominent_emotion = labels[scores.index(max(scores))]
        most_prominent_score = max(scores)

        st.write(f"Total Score: {total_score:.4f}")
        st.write(f"Most Prominent Emotion: {most_prominent_emotion} with a score of {most_prominent_score:.4f}")

        # Add to history
        st.session_state.history.append({
            'sentence': input_text,
            'labels': labels,
            'scores': scores,
            'total_score': total_score,
            'most_prominent_emotion': most_prominent_emotion,
            'most_prominent_score': most_prominent_score
        })
    else:
        st.warning("Please enter a sentence.")

# Display history
st.subheader("History")
for entry in st.session_state.history:
    st.write(f"Sentence: '{entry['sentence']}'")
    st.write("Predictions:")
    for label, score in zip(entry['labels'], entry['scores']):
        st.write(f"  {label}: {score:.4f}")
    st.write(f"Total Score: {entry['total_score']:.4f}")
    st.write(f"Most Prominent Emotion: {entry['most_prominent_emotion']} with a score of {entry['most_prominent_score']:.4f}")
    st.write("-" * 50)
