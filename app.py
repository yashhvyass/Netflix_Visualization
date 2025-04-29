# ---- Import libraries ----
import streamlit as st
st.set_page_config(
    page_title="Netflix Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import matplotlib

# ---- Load Netflix dataset ----
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    
    # --- Data preprocessing ---
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y', errors='coerce')
    
    df['month_added'] = df['date_added'].dt.month
    df['month_name_added'] = df['date_added'].dt.month_name()
    df['year_added'] = df['date_added'].dt.year
    
    # Add count column for aggregation
    df['count'] = 1
    
    # Extract first country
    df['first_country'] = df['country'].fillna('Unknown').apply(lambda x: x.split(',')[0].strip())
    
    # Fix missing/empty values
    df['first_country'].replace('', 'Unknown', inplace=True)
    df['first_country'].fillna('Unknown', inplace=True)

    # Replace long country names
    df['first_country'] = df['first_country'].replace('United States', 'USA')
    df['first_country'] = df['first_country'].replace('United Kingdom', 'UK')
    df['first_country'] = df['first_country'].replace('South Korea', 'S. Korea')
    
    # Create target age mapping
    ratings_ages = {
        'TV-PG': 'Older Kids',
        'TV-MA': 'Adults',
        'TV-Y7-FV': 'Older Kids',
        'TV-Y7': 'Older Kids',
        'TV-14': 'Teens',
        'R': 'Adults',
        'TV-Y': 'Kids',
        'NR': 'Adults',
        'PG-13': 'Teens',
        'TV-G': 'Kids',
        'PG': 'Older Kids',
        'G': 'Kids',
        'UR': 'Adults',
        'NC-17': 'Adults'
    }
    
    df['target_ages'] = df['rating'].replace(ratings_ages)
    
    # Process genres
    df['genre'] = df['listed_in'].apply(lambda x: x.replace(' ,',',').replace(', ',',').split(','))
    
    # Extract release year
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    
    return df

df = load_data()

# ---- Sidebar Navigation ----
st.sidebar.title("📺 Netflix Dashboard Navigation")
page = st.sidebar.radio("Select a Page:", ["Page 1: Evolution of Content Over the Years", 
                                           "Page 2: Netflix's Strategy", 
                                           "Page 3: Demographic Analysis", 
                                           "Page 4: Global Expansion", 
                                           "Page 5: Genre Analysis"])

# -------------------------------------------
# --------- PAGE 1: Content Trends ----------
# -------------------------------------------
# -------------------------------------------
# --------- PAGE 1: Content Trends ----------
# -------------------------------------------
if page == "Page 1: Evolution of Content Over the Years":
    st.title("📈 Netflix Content Trends Overview")

    st.markdown("---")
    # st.header("Netflix Content Overview")

    # --- Center Plot (Netflix Timeline) ---
    st.subheader("🗓️ Netflix Timeline (Major Milestones)")
    
    tl_dates = [
        "1997\nFounded", "1998\nMail Service", "2003\nGoes Public",
        "2007\nStreaming", "2016\nGlobal Launch", "2021\nNetflix & Chill"
    ]
    tl_x = [1, 2, 4, 5.3, 8, 9]

    tl_sub_x = [1.5, 3, 5, 6.5, 7]
    tl_sub_times = ["1998", "2000", "2006", "2010", "2012"]
    tl_text = [
        "Netflix.com launched", "Starts\nRecommendations",
        "Billionth DVD Delivery", "Canadian\nLaunch", "UK Launch"
    ]

    fig2, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.set_ylim(-2, 1.75)
    ax.set_xlim(0, 10)

    ax.axhline(0, xmin=0.1, xmax=0.9, c='#4a4a4a', zorder=1)
    ax.scatter(tl_x, np.zeros(len(tl_x)), s=120, c='#4a4a4a', zorder=2)
    ax.scatter(tl_x, np.zeros(len(tl_x)), s=30, c='#fafafa', zorder=3)
    ax.scatter(tl_sub_x, np.zeros(len(tl_sub_x)), s=50, c='#4a4a4a', zorder=4)

    for x, date in zip(tl_x, tl_dates):
        ax.text(x, -0.55, date, ha='center', fontfamily='serif', fontweight='bold', color='#4a4a4a', fontsize=10)

    levels = np.zeros(len(tl_sub_x))    
    levels[::2] = 0.3
    levels[1::2] = -0.3
    markerline, stemline, baseline = ax.stem(tl_sub_x, levels)
    plt.setp(baseline, zorder=0)
    plt.setp(markerline, marker=',', color='#4a4a4a')
    plt.setp(stemline, color='#4a4a4a')

    for idx, x, time, txt in zip(range(1, len(tl_sub_x)+1), tl_sub_x, tl_sub_times, tl_text):
        ax.text(x, 1.3*(idx%2)-0.5, time, ha='center', fontfamily='serif', fontweight='bold', color='#4a4a4a')
        ax.text(x, 1.3*(idx%2)-0.6, txt, va='top', ha='center', fontfamily='serif', color='#4a4a4a')

    ax.set_xticks([])
    ax.set_yticks([])

    fig2.suptitle("Netflix Timeline", fontsize=12, fontweight='bold', fontfamily='serif')

    st.pyplot(fig2)

    # ---- Horizontal Line Separator ----
    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # --- Bottom two columns ---
    col1, col2 = st.columns([5, 5])

    # --- Bottom left (col1) : Cumulative Additions ---
    with col1:
        st.subheader("📈 Cumulative Content Additions")

        data_sub = df.groupby('type')['year_added'].value_counts().unstack().fillna(0).loc[['TV Show', 'Movie']].cumsum(axis=0).T

        colors = {'Movie': '#b20710', 'TV Show': '#4a4a4a'}
        fig1 = go.Figure()

        for content_type in ['Movie', 'TV Show']:
            fig1.add_trace(go.Scatter(
                x=data_sub.index,
                y=data_sub[content_type],
                mode='lines',
                name=content_type,
                line=dict(color=colors[content_type], width=3),
            ))

        fig1.update_layout(
            title="<b>Evolution Over Time</b>",
            xaxis=dict(title="Year", tickmode='linear', dtick=1, range=[2008, 2020]),
            yaxis=dict(title="Cumulative Additions"),
            plot_bgcolor='white',
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig1, use_container_width=True)

    # --- Bottom right (col2) : Polar Plot ---
    with col2:
        st.subheader("🌀 Content Added by Month (Polar Plot)")

        data_month = df.groupby('month_name_added').size().reset_index(name='Value')
        data_month = data_month.sort_values(by='Value', ascending=False)

        color_map = ['#221f1f' for _ in range(12)]
        color_map[0] = color_map[-1] = '#b20710'

        fig3, ax = plt.subplots(figsize=(12,8), subplot_kw=dict(polar=True))  
        plt.axis('off')

        upperLimit = 30
        lowerLimit = 1
        labelPadding = 30

        max_value = data_month['Value'].max()
        slope = (max_value - lowerLimit) / max_value
        heights = slope * data_month.Value + lowerLimit

        width = 2 * np.pi / len(data_month.index)
        indexes = list(range(1, len(data_month.index)+1))
        angles = [element * width for element in indexes]

        bars = ax.bar(
            x=angles,
            height=heights,
            width=width,
            bottom=lowerLimit,
            linewidth=2,
            edgecolor="white",
            color=color_map,
            alpha=0.9
        )

        for bar, angle, height, label in zip(bars, angles, heights, data_month["month_name_added"]):
            rotation = np.rad2deg(angle)
            alignment = "left" if angle < np.pi/2 or angle > 3*np.pi/2 else "right"
            if alignment == "right":
                rotation += 180

            ax.text(
                x=angle,
                y=lowerLimit + bar.get_height() + labelPadding,
                s=label,
                ha=alignment,
                va='center',
                fontsize=8,
                fontfamily='serif',
                rotation=rotation,
                rotation_mode="anchor"
            )

        fig3.text(0.5, 0.95, "Content Added by Month", ha='center', fontsize=16, fontweight='bold', fontfamily='serif')

        st.pyplot(fig3)

# -------------------------------------------
# --------- PAGE 2: Netflix Content Analysis -
# -------------------------------------------
elif page == "Page 2: Netflix's Strategy":
    st.title("🎬 Tracing Netflix's Strategy Shifts")

    st.markdown("---")
    # st.header("Netflix Content Analysis")

    # Getting top countries for content
    top_countries = df.groupby('first_country')['count'].sum().sort_values(ascending=False).head(10).reset_index()

    # Creating heatmap data for target ages by country
    df_heatmap = df.loc[df['first_country'].isin(top_countries['first_country'])]
    df_heatmap = pd.crosstab(df_heatmap['first_country'], df_heatmap['target_ages'], normalize="index").T

    # Full-width columns
    col1, col2 = st.columns([5, 5])

    # --- Top left (col1) : Plot 1 ---
    with col1:
        st.subheader("🎥 Movie & TV Show Distribution")

        # Compute mf_ratio
        x = df.groupby(['type'])['type'].count()
        y = len(df)
        r = ((x/y)).round(2)
        mf_ratio = pd.DataFrame(r).T

        fig1, ax = plt.subplots(figsize=(12, 5))

        ax.barh(mf_ratio.index, mf_ratio['Movie'], color='#b20710', alpha=0.9, label='Movie')
        ax.barh(mf_ratio.index, mf_ratio['TV Show'], left=mf_ratio['Movie'], color='#221f1f', alpha=0.9, label='TV Show')

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        for i in mf_ratio.index:
            ax.annotate(f"{int(mf_ratio['Movie'][i]*100)}%", xy=(mf_ratio['Movie'][i]/2, i), va='center', ha='center', fontsize=30, color='white')
            ax.annotate("Movie", xy=(mf_ratio['Movie'][i]/2, -0.25), va='center', ha='center', fontsize=15, color='white')

            ax.annotate(f"{int(mf_ratio['TV Show'][i]*100)}%", xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, i), va='center', ha='center', fontsize=30, color='white')
            ax.annotate("TV Show", xy=(mf_ratio['Movie'][i]+mf_ratio['TV Show'][i]/2, -0.25), va='center', ha='center', fontsize=15, color='white')

        for s in ['top', 'left', 'right', 'bottom']:
            ax.spines[s].set_visible(False)

        fig1.text(0.1, 1.03, 'Movie & TV Show distribution', fontsize=15, fontweight='bold')
        fig1.text(0.1, 0.92, 'We see vastly more movies than TV shows on Netflix.', fontsize=12)

        st.pyplot(fig1)

    # --- Top right (col2) : Plot 2 ---
    with col2:
        st.subheader("⭐ Rating Distribution by Movies & TV Shows")

        # Compute rating distribution
        order = pd.DataFrame(df.groupby('rating').size().sort_values(ascending=False).reset_index())
        rating_order = list(order['rating'])
        mf = df.groupby('type')['rating'].value_counts().unstack().sort_index().fillna(0).astype(int)[rating_order]

        movie = mf.loc['Movie']
        tv = - mf.loc['TV Show']

        fig2, ax = plt.subplots(figsize=(12,6))
        ax.bar(movie.index, movie, width=0.5, color='#b20710', alpha=0.9)
        ax.bar(tv.index, tv, width=0.5, color='#221f1f', alpha=0.9)

        for i in tv.index:
            ax.annotate(f"{-tv[i]}", xy=(i, tv[i] - 60), ha='center', fontsize=10, color='#4a4a4a')

        for i in movie.index:
            ax.annotate(f"{movie[i]}", xy=(i, movie[i] + 60), ha='center', fontsize=10, color='#4a4a4a')

        for s in ['top', 'left', 'right', 'bottom']:
            ax.spines[s].set_visible(False)

        ax.set_xticklabels(mf.columns, fontfamily='serif', rotation=45, ha='right')
        ax.set_yticks([])

        fig2.text(0.16, 1.03, 'Rating distribution by Film & TV Show', fontsize=15, fontweight='bold')
        fig2.text(0.16, 0.92, '''We observe that some ratings are only applicable to Movies. The most common for both Movies & TV Shows are TV-MA and TV-14.''', fontsize=12)

        fig2.text(0.75,1.0,"Movie", fontweight="bold", fontsize=15, color='#b20710')
        fig2.text(0.81,1.0,"|", fontweight="bold", fontsize=15, color='black')
        fig2.text(0.82,1.0,"TV Show", fontweight="bold", fontsize=15, color='#221f1f')

        st.pyplot(fig2)

    # ---- Horizontal Line Separator ----
    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # Full-width bottom columns
    col3, col4 = st.columns([5, 5])

    # --- Bottom left (col3) : Plot 3 ---
    with col3:
        st.subheader("🎯Target Audience Heatmap")

        # Define orders for consistent display
        country_order = top_countries['first_country'].head(9).tolist()
        age_order = ['Kids', 'Older Kids', 'Teens', 'Adults']

        # Get heatmap data
        heatmap_data = df_heatmap.loc[age_order, country_order]

        fig3 = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=country_order,
            y=age_order,
            colorscale='Reds',
            zmin=0.05,
            zmax=0.6,
            colorbar=dict(
                title=dict(text="Proportion", font=dict(family='serif', color='black', size=14)), 
            ),
            text=[[f'{v:.0%}' for v in row] for row in heatmap_data.values],
            texttemplate="%{text}",
            textfont={"size":12, "family":"serif", "color":"black"},
            hovertemplate='Age Group: %{y}<br>Country: %{x}<br>Proportion: %{z:.0%}<extra></extra>'
        ))

        # Add cell borders
        shapes = []
        n_cols = len(country_order)
        n_rows = len(age_order)
        cell_width = 1 / n_cols
        cell_height = 1 / n_rows

        for i in range(n_cols):
            for j in range(n_rows):
                shapes.append(
                    dict(
                        type="rect",
                        xref="paper", yref="paper",
                        x0=i * cell_width,
                        y0=1 - (j + 1) * cell_height,
                        x1=(i + 1) * cell_width,
                        y1=1 - j * cell_height,
                        line=dict(color="black", width=1),
                        fillcolor="rgba(0,0,0,0)"
                    )
                )

        # Final layout update including axis label colors
        fig3.update_layout(
            shapes=shapes,
            title="<b>Target Ages Proportion of Content by Country</b>",
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                tickfont=dict(family='serif', color='black', size=12)
            ),
            yaxis=dict(
                tickfont=dict(family='serif', color='black', size=12)
            )
        )

        st.plotly_chart(fig3, use_container_width=True)

    # --- Bottom right (col4) : Plot 4 ---
    with col4:
        st.subheader("📊 Netflix Content WordCloud")
        # Custom color map (Netflix palette)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#221f1f', '#b20710'])

        # Create the text
        text = str(list(df['title'])).replace(',', '').replace('[', '').replace("'", '').replace(']', '').replace('.', '')

        # Create the mask
        mask = np.array(Image.open('netflix_01.webp'))

        # Generate the wordcloud with BORDER
        wordcloud = WordCloud(
            background_color='white', 
            width=2000, 
            height=1500,
            colormap=cmap, 
            max_words=500, 
            mask=mask,
            contour_color='black',        # <<< BLACK OUTLINE
            contour_width=3               # <<< OUTLINE THICKNESS
        ).generate(text)

        # Get the wordcloud as an array
        wordcloud_array = wordcloud.to_array()

        # Create a Plotly figure
        fig4 = go.Figure()

        fig4.add_trace(
            go.Image(z=wordcloud_array)
        )

        fig4.update_layout(
            margin=dict(l=10, r=10, t=80, b=10),
            xaxis_showgrid=False, 
            yaxis_showgrid=False,
            xaxis_showticklabels=False, 
            yaxis_showticklabels=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        fig4.update_traces(hoverinfo='skip')  # optional: disable hover labels over the image
        st.plotly_chart(fig4, use_container_width=True)


        # st.subheader("📅 Content Release Year Distribution")
        
        # # Create histogram of release years
        # fig4 = px.histogram(
        #     df,
        #     x='release_year',
        #     color='type',
        #     nbins=20,
        #     color_discrete_map={'Movie': '#b20710', 'TV Show': '#221f1f'},
        #     opacity=0.8,
        #     barmode='overlay',
        #     title='Distribution of Content by Release Year'
        # )
        
        # fig4.update_layout(
        #     xaxis_title="Release Year",
        #     yaxis_title="Number of Titles",
        #     legend_title="Content Type",
        #     plot_bgcolor='white'
        # )
        
        # st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------
# --------- PAGE 3: Country Analysis --------
# -------------------------------------------
# -------------------------------------------
# --------- PAGE 3: Country Analysis --------
# -------------------------------------------
elif page == "Page 3: Demographic Analysis":
    st.title("🌎 Netflix Content Distribution Trends")

    st.markdown("---")
    # st.header("Netflix Country-Focused Analysis")

    fig1 = go.Figure()

    # # --- Plot 1: Top 10 Countries ---
    st.subheader("🏳️ Top 10 Content Producing Countries")

    data = df.groupby('first_country')['count'].sum().sort_values(ascending=False)[:10]

    color_map = ['gray' for _ in range(10)]
    color_map[0] = color_map[1] = color_map[2] = '#b20710'  # Top 3 countries highlighted

    fig1.add_trace(go.Bar(
        x=data.index,
        y=data.values,
        marker_color=color_map,
        text=data.values,
        textposition='outside',
        hovertemplate='Country: %{x}<br>Count: %{y}<extra></extra>',
    ))

    fig1.update_layout(
        # title={
        #     'text': 'Top 10 Countries on Netflix',
        #     'y': 0.95,
        #     'x': 0.05,
        #     'xanchor': 'left',
        #     'yanchor': 'top',
        #     'font': dict(size=22, family='serif', color='black')
        # },
        # barmode='stack',
        # title=dict(
        #     text='<b>Top 10 Countries Movie & TV Show Split<br><sup>Percent Stacked Bar Chart</sup></b>',
        #     x=0.05,
        #     font=dict(family='serif', size=25)
        # ),
        annotations=[
            dict(
                text="The three most frequent countries have been highlighted.",
                x=0.15,
                y=0.87,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, family="serif", color="black")
            ),
            # dict(
            #     text="<b>Insights</b>",
            #     x=0.90,
            #     y=1.0,
            #     xref="paper",
            #     yref="paper",
            #     showarrow=False,
            #     font=dict(size=18, family="serif", color="black")
            # ),
            dict(
            text=("<b>Insights</b><br><br>"
                  "The most prolific producers of<br>"
                  "content for Netflix are primarily<br>"
                  "the USA, India, and the UK.<br><br>"
                  "Netflix being a US company explains<br>"
                  "the dominance of US content."),
            x=1.00,
            y=0.8,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=14, family="serif", color="black"),
            borderpad=10,
            bordercolor='black',
            borderwidth=1,
            )
        ],
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            tickfont=dict(size=12, family='serif'),
        ),
        yaxis=dict(
            visible=False,
        ),
        margin=dict(l=60, r=200, t=100, b=60),
        bargap=0.2,
    )

    fig1.update_xaxes(title_text='Country')

    st.plotly_chart(fig1, use_container_width=True)

    # ---- Horizontal Line Separator ----
    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)


    #########################
    ########################
    # --- Plot 2: Movie vs TV Show Split ---
    st.subheader("🎬 Top 10 Countries Movies vs TV Show Split")

    country_order = df['first_country'].value_counts()[:11].index
    data_q2q3 = df[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[country_order]
    data_q2q3['sum'] = data_q2q3.sum(axis=1)
    data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']].sort_values(by='Movie', ascending=False)[::-1]

    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        y=data_q2q3_ratio.index,
        x=data_q2q3_ratio['Movie'],
        name='Movie',
        orientation='h',
        marker=dict(color='#b20710'),
        text=[f"{v*100:.1f}%" for v in data_q2q3_ratio['Movie']],
        textposition='inside',
        insidetextanchor='middle'
    ))

    fig2.add_trace(go.Bar(
        y=data_q2q3_ratio.index,
        x=data_q2q3_ratio['TV Show'],
        name='TV Show',
        orientation='h',
        marker=dict(color='#221f1f'),
        text=[f"{v*100:.1f}%" for v in data_q2q3_ratio['TV Show']],
        textposition='inside',
        insidetextanchor='middle'
    ))

    fig2.update_layout(
        barmode='stack',
        # title=dict(
        #     # text='<b>Top 10 Countries Movie & TV Show Split<br><sup>Percent Stacked Bar Chart</sup></b>',
        #     x=0.05,
        #     font=dict(family='serif', size=25, color='black')
        # ),
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[0, 1]
        ),
        yaxis=dict(
            tickfont=dict(family='serif', size=12, color='black')
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(family='serif', size=14, color='black')
        ),
        margin=dict(l=60, r=300, t=100, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    fig2.add_annotation(
        x=1.40,
        y=1,
        xref='paper',
        yref='paper',
        showarrow=False,
        align='left',
        text=(
            "<b>Insights</b><br><br>"
            "India is dominated by Movies,<br>"
            "while South Korea focuses heavily<br>"
            "on TV Shows."
        ),
        font=dict(family='serif', size=14, color='black'),
        bordercolor='black',
        borderwidth=1,
        opacity=1,
        borderpad=10,
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------
# --------- PAGE 4: Ratings Analysis --------
# -------------------------------------------
# -------------------------------------------
# --------- PAGE 4: Netflix Expansion Map ----------
# -------------------------------------------
elif page == "Page 4: Global Expansion":
    st.title("🌍 Netflix Content Distribution")

    st.markdown("---")
    # st.header("Interactive World Map of Netflix's Content Expansion")

    # # Drop missing dates
    # df = df.dropna(subset=['year_added'])

    # Expand and clean country info
    countries_expanded = df.dropna(subset=['country']).copy()
    countries_expanded['primary_country'] = countries_expanded['country'].apply(lambda x: x.split(',')[0].strip())

    # Group by country
    country_counts = countries_expanded.groupby('primary_country').size().reset_index(name='count')
    country_counts = country_counts.rename(columns={'primary_country': 'country'})

    # Plotly Choropleth
    fig_map = px.choropleth(
        country_counts,
        locations='country',
        locationmode='country names',
        color='count',
        color_continuous_scale='OrRd',
        # title='🌎 Netflix Content Library: Global Expansion',
        range_color=(country_counts['count'].min(), country_counts['count'].max())
    )

    fig_map.update_geos(
        showcountries=True, countrycolor="black",
        showocean=True, oceancolor="lightblue",
        projection_type="natural earth"
    )

    fig_map.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0}
    )

    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------------------
# --------- PAGE 5: Genre Analysis ----------
# -------------------------------------------
elif page == "Page 5: Genre Analysis":
    st.title("🎭 Netflix Genre Connections")

    st.markdown("---")

    # ==================================================
    # ----------------- First Plot ---------------------
    # ==================================================
    st.header("🎯 Genre Correlation Heatmap")

    # Preprocess 'listed_in' to split genres
    df['genre_list'] = df['listed_in'].fillna('Unknown').apply(lambda x: [genre.strip() for genre in x.split(',')])

    # Create binary matrix
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(mlb.fit_transform(df['genre_list']), columns=mlb.classes_, index=df.index)

    # Find the Top 20 most common genres
    genre_counts = genre_matrix.sum().sort_values(ascending=False)
    top_20_genres = genre_counts.head(20).index.tolist()

    # Reorder and filter
    genre_matrix_top20 = genre_matrix[top_20_genres]
    genre_corr_top20 = genre_matrix_top20.corr()

    # Mask upper triangle
    import numpy as np
    mask = np.triu(np.ones_like(genre_corr_top20, dtype=bool))
    genre_corr_masked = genre_corr_top20.mask(mask)
    genre_corr_masked_filled = genre_corr_masked.fillna(0)

    # Create heatmap
    import plotly.graph_objects as go
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=genre_corr_masked_filled.values,
        x=top_20_genres,
        y=top_20_genres,
        colorscale='RdBu',
        zmin=-0.3,
        zmax=0.3,
        colorbar=dict(title='Correlation'),
        hoverongaps=False
    ))

    # Add black grid lines
    n = len(top_20_genres)
    for i in range(n+1):
        fig_heatmap.add_shape(type="line",
                              x0=i-0.5, y0=-0.5, x1=i-0.5, y1=n-0.5,
                              line=dict(color="black", width=1))
        fig_heatmap.add_shape(type="line",
                              x0=-0.5, y0=i-0.5, x1=n-0.5, y1=i-0.5,
                              line=dict(color="black", width=1))

    # Layout update
    fig_heatmap.update_layout(
        width=800,
        height=800,
        xaxis_title="Genre",
        yaxis_title="Genre",
        xaxis_tickangle=45,
        xaxis_side="bottom",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor='white',
        yaxis_autorange='reversed'
    )

    # Display first plot
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ------------------------------------------------
    # ---- Horizontal Separator ----
    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    # ==================================================
    # ----------------- Second Plot --------------------
    # ==================================================
    st.header("🎬 Netflix Genre Diversification Over Time (Sunburst)")

    # Expand genres
    genres_expanded = df.dropna(subset=['listed_in']).copy()
    genres_expanded['genre'] = genres_expanded['listed_in'].str.split(', ')
    genres_expanded = genres_expanded.explode('genre')

    # Group by year and genre
    genre_counts = genres_expanded.groupby(['year_added', 'genre']).size().reset_index(name='count')

    # Create Sunburst Chart
    import plotly.express as px
    fig3 = px.sunburst(
        genre_counts,
        path=['year_added', 'genre'],
        values='count',
        # title='🎬 Netflix Genre Diversification Over Time',
        width=1000,
        height=700,
        color_discrete_sequence=['#E50914', '#221f1f', '#B81D24', '#ff758f', '#000000', '#087e8b', '#004e89']
    )

    fig3.update_layout(
        margin=dict(t=50, l=0, r=0, b=50),
        plot_bgcolor='white',           # White background
        paper_bgcolor='white',           # Paper background white
        font_color='black',              # Black font color
        title_font_size=24,
        title_x=0.5                      # Center title
    )

    fig3.update_traces(
        insidetextfont=dict(color='#f5f3f4'),  # Make inside text lighter
        selector=dict(type='sunburst')
    )

    # Display second plot
    st.plotly_chart(fig3, use_container_width=True)

    
