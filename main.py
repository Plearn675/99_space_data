import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the visual style
sns.set_theme(style="whitegrid")


def run_analysis():
    # 1. LOAD DATA
    try:
        # Note: I used 'data/mission_launches.csv' as per your previous script
        df = pd.read_csv('data/mission_launches.csv')
        print("✅ Data loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: mission_launches.csv not found. Check the /data folder.")
        return

    # 2. DATA CLEANING
    df_clean = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], errors='ignore')
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], utc=True, errors='coerce')
    df_clean = df_clean.dropna(subset=['Date'])

    df_clean['Year'] = df_clean['Date'].dt.year
    df_clean['Month'] = df_clean['Date'].dt.month
    df_clean['Decade'] = (df_clean['Year'] // 10) * 10

    # Create the 5-year "Bucket" column for grouping
    df_clean['5yr_Period'] = (df_clean['Year'] // 5) * 5

    df_clean['Price'] = pd.to_numeric(df_clean['Price'].astype(str).str.replace(',', ''), errors='coerce')
    df_clean.to_csv('cleaned_mission_launches.csv', index=False)

    # 3. CALCULATIONS
    # Q1: Yearly Cost
    yearly_cost = df_clean.dropna(subset=['Price']).groupby('Year')['Price'].mean()

    # Q2: Popular Months
    monthly_counts = df_clean['Month'].value_counts().sort_index()
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    monthly_counts.index = monthly_counts.index.map(month_names)

    # Q3: Success Rate
    safety_stats = df_clean.groupby(['Decade', 'Mission_Status']).size().unstack(fill_value=0)
    safety_stats['Success_Rate'] = (safety_stats['Success'] / safety_stats.sum(axis=1)) * 100

    # Q4: Dominant Org every 5 years
    five_year_counts = df_clean.groupby(['5yr_Period', 'Organisation']).size().reset_index(name='Launches')
    dominant_5yr = five_year_counts.loc[five_year_counts.groupby('5yr_Period')['Launches'].idxmax()]

    # 4. VISUALIZATION
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Monthly Popularity
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, ax=axes[0, 0], palette='magma')
    axes[0, 0].set_title('Popularity of Months for Launches', fontsize=14)

    # Plot 2: Average Cost over Time
    sns.lineplot(x=yearly_cost.index, y=yearly_cost.values, marker='o', ax=axes[0, 1], color='blue')
    axes[0, 1].set_title('Average Mission Cost by Year (USD Millions)', fontsize=14)

    # Plot 3: Success Rate Improvement
    sns.lineplot(x=safety_stats.index, y=safety_stats['Success_Rate'], marker='s', ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Safety Trend: Success Rate by Decade', fontsize=14)
    axes[1, 0].set_ylim(0, 105)

    # Plot 4: Dominant Organization every 5 Years
    sns.barplot(x='5yr_Period', y='Launches', hue='Organisation', data=dominant_5yr, ax=axes[1, 1], dodge=False)
    axes[1, 1].set_title('Dominant Organization (5-Year Intervals)', fontsize=14)
    axes[1, 1].set_xlabel('Period Starting In...')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('space_analysis_dashboard.png')
    print("✅ Dashboard saved as 'space_analysis_dashboard.png'.")
    plt.show()


if __name__ == "__main__":
    run_analysis()