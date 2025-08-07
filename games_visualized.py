import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def table_foul_stats(df):
    home_games = df[df['Status'] == 'Home']
    away_games = df[df['Status'] == 'Away']
    
    selected_columns = ['Fouls Committed', 'Fouls Won', 'Yellow Cards', 'Red Cards'] 
    grouped_stats_home = home_games.groupby('Team')[selected_columns].mean()
    grouped_stats_away = away_games.groupby('Team')[selected_columns].mean()
    
    # Uncomment one function at a time.

    # display_fouls_won(grouped_stats_home, grouped_stats_away)
    # display_fouls_committed(grouped_stats_home, grouped_stats_away)
    # display_yellow_cards(grouped_stats_home, grouped_stats_away)
    # display_yellow_cards_distribution(grouped_stats_home, grouped_stats_away)
    # display_red_cards(grouped_stats_home, grouped_stats_away)
    # display_avg_foulstats(grouped_stats_home, grouped_stats_away)

def display_fouls_won(home, away):
    teams = home.index
    
    fouls_won_home = home['Fouls Won']
    fouls_won_away = away['Fouls Won']

    bar_width = 0.35
    r1 = range(len(teams))
    r2 = [x + bar_width for x in r1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(r1, fouls_won_home, color='green', width=bar_width, edgecolor='black', label='Fouls Won (Home)')
    ax.bar(r2, fouls_won_away, color='blue', width=bar_width, edgecolor='black', label='Fouls Won (Away)')
    
    ax.set_xlabel('Teams', fontweight='bold')
    ax.set_ylabel('Fouls Won', fontweight='bold')
    ax.set_title('Average Fouls Won: Home vs. Away', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in range(len(teams))])
    ax.set_xticklabels(teams, rotation=90)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def display_fouls_committed(home, away):
    teams = home.index
    
    fouls_committed_home = home['Fouls Committed']
    fouls_committed_away = away['Fouls Committed']

    bar_width = 0.35
    r1 = range(len(teams))
    r2 = [x + bar_width for x in r1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(r1, fouls_committed_home, color='green', width=bar_width, edgecolor='black', label='Fouls Committed (Home)')
    ax.bar(r2, fouls_committed_away, color='blue', width=bar_width, edgecolor='black', label='Fouls Committed (Away)')
    
    ax.set_xlabel('Teams', fontweight='bold')
    ax.set_ylabel('Fouls Committed', fontweight='bold')
    ax.set_title('Average Fouls Committed: Home vs. Away', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in range(len(teams))])
    ax.set_xticklabels(teams, rotation=90)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def display_yellow_cards(home, away):
    teams = home.index
    
    yellow_cards_home = home['Yellow Cards']
    yellow_cards_away = away['Yellow Cards']

    bar_width = 0.35
    r1 = range(len(teams))
    r2 = [x + bar_width for x in r1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(r1, yellow_cards_home, color='green', width=bar_width, edgecolor='black', label='Yellow Cards (Home)')
    ax.bar(r2, yellow_cards_away, color='blue', width=bar_width, edgecolor='black', label='Yellow Cards (Away)')
    
    ax.set_xlabel('Teams', fontweight='bold')
    ax.set_ylabel('Yellow Cards', fontweight='bold')
    ax.set_title('Average Yellow Cards: Home vs. Away', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in range(len(teams))])
    ax.set_xticklabels(teams, rotation=90)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def display_yellow_cards_distribution(home, away):
    yellow_cards_home = home['Yellow Cards']
    yellow_cards_away = away['Yellow Cards']

    plt.figure(figsize=(10, 6))
    sns.kdeplot(yellow_cards_home, color='green', shade=True, label='Yellow Cards (Home)')
    sns.kdeplot(yellow_cards_away, color='blue', shade=True, label='Yellow Cards (Away)')
    
    plt.xlabel('Yellow Cards', fontweight='bold')
    plt.ylabel('Density', fontweight='bold')
    plt.title('Distribution of Yellow Cards: Home vs. Away', fontweight='bold')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def display_red_cards(home, away):
    teams = home.index
    
    red_cards_home = home['Red Cards']
    red_cards_away = away['Red Cards']

    bar_width = 0.35
    r1 = range(len(teams))
    r2 = [x + bar_width for x in r1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(r1, red_cards_home, color='green', width=bar_width, edgecolor='black', label='Red Cards (Home)')
    ax.bar(r2, red_cards_away, color='blue', width=bar_width, edgecolor='black', label='Red Cards (Away)')
    
    ax.set_xlabel('Teams', fontweight='bold')
    ax.set_ylabel('Red Cards', fontweight='bold')
    ax.set_title('Average Red Cards: Home vs. Away', fontweight='bold')
    ax.set_xticks([r + bar_width / 2 for r in range(len(teams))])
    ax.set_xticklabels(teams, rotation=90)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def display_avg_foulstats(home, away):
    overall_avg_home = home.mean(axis=0)
    overall_avg_away = away.mean(axis=0)
    
    stats = ['Fouls Committed', 'Fouls Won', 'Yellow Cards', 'Red Cards']
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stats))
    home_bars = ax.bar(x - bar_width/2, overall_avg_home, width=bar_width, label='Home', edgecolor='black', color='green')
    away_bars = ax.bar(x + bar_width/2, overall_avg_away, width=bar_width, label='Away', edgecolor='black', color='blue')
    
    ax.set_ylabel('Averages')
    ax.set_title('Average Foul Stats (Home vs Away)')
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.legend()

    for bar in home_bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), round(bar.get_height(), 2),
                ha='center', va='bottom', color='black', fontsize=10)
    for bar in away_bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), round(bar.get_height(), 2),
                ha='center', va='bottom', color='black', fontsize=10)
    plt.show()

def table_result_stats(df):
    selected_columns = ['Score', 'Possession Percentage', 'Total Shots', 'Shots on Target', 'Passing Percentage', 'Result']
    selected_df = df[selected_columns]
    result_stats = selected_df.groupby('Result').mean()
    result_stats = result_stats.reindex(['Win', 'Draw', 'Lose'])
    display_result_stats(result_stats)

def display_result_stats(results):
    columns = ['Score', 'Possession Percentage', 'Total Shots', 'Shots on Target', 'Passing Percentage']
    states = results.index
    bar_width = 0.25
    x = np.arange(len(columns))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, state in enumerate(states):
        bars = ax.bar(x + i * bar_width, results.loc[state], width=bar_width, label=state, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_xlabel('Game Metrics')
    ax.set_ylabel('Averages')
    ax.set_title('Average Statistics by Result')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(columns)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def model(df):
    sns.pairplot(df[['Possession Percentage', 'Total Shots', 'Shots on Target', 'Passing Percentage', 'Result']], hue='Result')
    plt.show()

    # Feature selection
    X = df[['Possession Percentage', 'Total Shots', 'Shots on Target', 'Passing Percentage']]
    y = df['Result']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Model Evaluation
    y_pred = rf_classifier.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature Importance
    feature_importance = (pd.Series(rf_classifier.feature_importances_, index=X.columns)).nlargest(4) 
    print("Feature Importance:\n", feature_importance)

def main():
    df = pd.read_csv('spiders\games.csv')

    # Uncomment one at a time, 'table_foul_stats(df)' has 6 potentinal functions to Uncomment.

    # table_foul_stats(df)
    # table_result_stats(df)
    # model(df)

if __name__ == '__main__':
    main()