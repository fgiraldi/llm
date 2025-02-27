import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


def create_trend_visualizations(analysis_results):
    """
    Create interactive visualizations for review analysis results

    Parameters:
    analysis_results: Dictionary containing analysis results from analyze_review_trends()

    Returns:
    dict: Dictionary of Plotly figures
    """
    # Convert monthly trends to DataFrame
    monthly_df = pd.DataFrame(analysis_results['monthly_trends'])

    # 1. Rating and Cluster Trends Over Time
    fig_trends = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Rating Over Time', 'Dominant Clusters Over Time'),
        vertical_spacing=0.15
    )

    # Add rating trend
    fig_trends.add_trace(
        go.Scatter(
            x=monthly_df['date'],
            y=monthly_df['rating'],
            mode='lines+markers',
            name='Average Rating',
            line=dict(color='#2E86C1'),
            hovertemplate='Date: %{x}<br>Rating: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add cluster trend
    fig_trends.add_trace(
        go.Scatter(
            x=monthly_df['date'],
            y=monthly_df['cluster'],
            mode='lines+markers',
            name='Dominant Cluster',
            line=dict(color='#E67E22'),
            hovertemplate='Date: %{x}<br>Cluster: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    fig_trends.update_layout(
        height=700,
        showlegend=True,
        title_text="Review Trends Over Time",
        title_x=0.5
    )

    # 2. Cluster Term Heatmap
    terms_data = []
    for cluster, data in analysis_results['cluster_terms'].items():
        for i, term in enumerate(data['terms']):
            terms_data.append({
                'Cluster': cluster,
                'Term': term,
                'Importance': (len(data['terms']) - i) / len(data['terms']),
                'Cluster Size': data['size']
            })

    terms_df = pd.DataFrame(terms_data)
    fig_heatmap = px.density_heatmap(
        terms_df,
        x='Cluster',
        y='Term',
        z='Importance',
        title='Top Terms by Cluster',
        color_continuous_scale='Viridis'
    )

    # Add cluster size information to hover
    fig_heatmap.update_traces(
        hovertemplate='Cluster: %{x}<br>' +
                      'Term: %{y}<br>' +
                      'Importance: %{z:.2f}<br>' +
                      'Cluster size: %{customdata}<extra></extra>',
        customdata=[
            [size] * len(data['terms'])
            for cluster, data in analysis_results['cluster_terms'].items()
            for size in [data['size']]
        ]
    )

    fig_heatmap.update_layout(
        height=600,
        title_x=0.5
    )

    # 3. Rating Distribution by Cluster
    category_clusters = pd.DataFrame(analysis_results['category_clusters'])
    fig_category = px.bar(
        category_clusters,
        title='Review Distribution by Category and Cluster',
        barmode='group',
        height=500
    )

    fig_category.update_layout(
        xaxis_title='Category',
        yaxis_title='Number of Reviews',
        title_x=0.5,
        showlegend=True,
        legend_title='Cluster'
    )

    return {
        'trends': fig_trends,
        'terms': fig_heatmap,
        'categories': fig_category
    }


def save_visualizations(figures, output_dir='./visualizations'):
    """
    Save visualizations as HTML files

    Parameters:
    figures: Dictionary of Plotly figures
    output_dir: Directory to save the files
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each figure
    for name, fig in figures.items():
        fig.write_html(f"{output_dir}/review_{name}.html")


def print_cluster_summary(analysis_results):
    """
    Print a human-readable summary of cluster sizes and their top terms
    """
    print("\nCluster Analysis Summary:")
    print("=" * 50)

    for cluster, data in analysis_results['cluster_terms'].items():
        print(f"\n{cluster}:")
        print(f"Size: {data['size']} documents ({data['percentage']:.1f}% of total)")
        print("Top terms:", ", ".join(data['terms'][:5]))

    print("\nTotal documents analyzed:", sum(data['size'] for data in analysis_results['cluster_terms'].values()))


def visualize_cluster_sizes(analysis_results):
    """
    Create a visualization of cluster sizes with their top terms
    """
    clusters = []
    sizes = []
    percentages = []
    top_terms = []

    for cluster, data in analysis_results['cluster_terms'].items():
        clusters.append(cluster)
        sizes.append(data['size'])
        percentages.append(data['percentage'])
        top_terms.append("<br>".join([
            f"• {term}" for term in data['terms'][:5]
        ]))

    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            x=clusters,
            y=sizes,
            text=[f"{p:.1f}%" for p in percentages],
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>" +
                         "Documents: %{y}<br>" +
                         "Percentage: %{text}<br>" +
                         "Top terms:<br>%{customdata}" +
                         "<extra></extra>",
            customdata=top_terms
        )
    ])

    fig.update_layout(
        title="Cluster Sizes and Top Terms",
        title_x=0.5,
        xaxis_title="Clusters",
        yaxis_title="Number of Documents",
        height=500,
        showlegend=False,
        hoverlabel={'align': 'left'}
    )

    fig.write_html("./visualizations/cluster_sizes.html")
