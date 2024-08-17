import streamlit as st
from transformers import pipeline
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# Initialize the text classification pipeline with the specified model
try:
    classifier = pipeline(task="text-classification", model=r"JAGADEESH51/TEXT_EMOTION_RECOGNITION", top_k=None)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Store previous sentences and their emotion scores
if 'history' not in st.session_state:
    st.session_state.history = []

# Set a threshold for emotion scores
if 'score_threshold' not in st.session_state:
    st.session_state.score_threshold = 0.1

# Function to classify sentences and update previous feeds
def classify_and_update(sentences):
    all_sentences = [entry['sentence'] for entry in st.session_state.history] + sentences
    try:
        model_outputs = classifier(all_sentences)
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return []

    new_results = []
    for idx, output in enumerate(model_outputs[len(st.session_state.history):]):
        sentence = all_sentences[len(st.session_state.history) + idx]
        filtered_emotions = [emotion for emotion in output if emotion['score'] >= st.session_state.score_threshold]
        new_results.append({"sentence": sentence, "emotions": filtered_emotions})

    return new_results

# Function to create an interactive bar chart
def plot_emotion_scores(labels, scores, sentence):
    fig = go.Figure()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#F0F0F0']
    fig.add_trace(go.Bar(
        x=labels,
        y=scores,
        text=[f'{score:.2f}' for score in scores],
        textposition='outside',
        marker=dict(color=colors[:len(labels)]),
        name='Emotion Scores'
    ))

    fig.update_layout(
        title=f'Emotion Scores for Sentence: "{sentence}"',
        xaxis_title='Emotions',
        yaxis_title='Scores',
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=12)
    )
    fig.update_traces(marker=dict(line=dict(color='black', width=1.5)))
    st.plotly_chart(fig, use_container_width=True)

# Function to create a radar chart
def plot_radar_chart(labels, scores, sentence):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name=f'Sentence: "{sentence}"'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title=f'Radar Chart for Sentence: "{sentence}"',
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        font=dict(size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to create a refined word cloud
def plot_word_cloud(emotions):
    text = ' '.join(emotions)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    st.write("### Emotion Word Cloud")
    st.write("The word cloud visualizes the most frequently occurring emotions in the analyzed sentences. Larger words represent emotions that appear more frequently, providing a quick overview of predominant feelings.")
    st.image(buf, caption='Word Cloud of Emotions', use_column_width=True)

# Function to create a pie chart
def plot_pie_chart(labels, scores):
    fig = go.Figure(data=[go.Pie(labels=labels, values=scores, hole=0.3)])
    fig.update_layout(title='Emotion Distribution', margin=dict(l=50, r=50, t=50, b=50), font=dict(size=12))
    st.plotly_chart(fig, use_container_width=True)

# Streamlit app interface
st.title("Emotion Analysis App")

st.write("""
    **Welcome to the Emotion Analysis App!** 
    Enter sentences to analyze the emotions expressed. 
    You can see a detailed breakdown of emotions, visualize them with charts and a word cloud, and review previous inputs and results.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select an option", ["Analyze Sentences", "View History"], key='navigation')

# Set threshold with slider
score_threshold = st.sidebar.slider("Select Threshold for Emotion Scores", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
st.session_state.score_threshold = score_threshold

# Analyze sentences section
if page == "Analyze Sentences":
    user_input = st.text_area("Enter sentences to analyze:", "")

    if st.button("Analyze"):
        if user_input:
            sentences = user_input.split('\n')
            results = classify_and_update(sentences)

            if results:
                # Store results to avoid repetition on button click
                if 'analyzed_results' not in st.session_state:
                    st.session_state.analyzed_results = results
                else:
                    st.session_state.analyzed_results.extend(results)

                # Display all visuals (Bar, Pie, Radar, Word Cloud)
                for result in results:
                    sentence = result["sentence"]
                    emotions = result["emotions"]

                    if not emotions:
                        st.write(f"No significant emotions detected for the sentence: '{sentence}'")
                    else:
                        labels = [emotion['label'] for emotion in emotions]
                        scores = [emotion['score'] for emotion in emotions]

                        st.write(f"**Sentence:** '{sentence}'")
                        st.write(f"**Predictions:** {[(label, f'{score:.2f}') for label, score in zip(labels, scores)]}")

                        # Plot all visualizations
                        plot_emotion_scores(labels, scores, sentence)
                        plot_pie_chart(labels, scores)
                        plot_radar_chart(labels, scores, sentence)
                        plot_word_cloud(labels)

                        # Add new results to history
                        st.session_state.history.append({"input": f"Input {len(st.session_state.history) + 1}", "sentence": sentence, "emotions": emotions})
            else:
                st.write("No results to display.")
        else:
            st.error("Please enter at least one sentence to analyze.")

# View history section
elif page == "View History":
    if st.button("Clear History"):
        st.session_state.history = []
        st.write("History cleared.")

    if st.session_state.history:
        st.write("**Previous Inputs and Results:**")

        # Option to view more visuals
        show_more_visuals = st.button("View More Visuals")

        for entry in st.session_state.history:
            st.write(f"**{entry['input']} - Sentence:** '{entry['sentence']}'")
            emotions = entry["emotions"]
            if emotions:
                labels = [emotion['label'] for emotion in emotions]
                scores = [emotion['score'] for emotion in emotions]

                st.write(f"**Predictions:** {[(label, f'{score:.2f}') for label, score in zip(labels, scores)]}")

                # Always show bar and pie charts
                plot_emotion_scores(labels, scores, entry["sentence"])
                plot_pie_chart(labels, scores)

                if show_more_visuals:
                    # Show radar chart and word cloud when "View More Visuals" is clicked
                    plot_radar_chart(labels, scores, entry["sentence"])
                    plot_word_cloud(labels)
    else:
        st.write("No history available.")
