import pandas as pd
import json
import os

def generate_html_dashboard(csv_path='results/valid_scenarios.csv', output_path='results/dashboard.html'):
    """
    Reads the challenge results and generates a beautiful HTML dashboard 
    with Chart.js and a premium dark UI.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run main.py first.")
        return

    # Load data
    df = pd.read_csv(csv_path).head(20)
    
    # Prepare data for JSON
    labels = [f"Scenario {i+1}" for i in range(len(df))]
    p80_data = df['P80'].tolist()
    r95_data = df['R95'].tolist()
    table_html = df.to_html(classes='data-table', index=False)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Asteroid Challenge Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {{
                --primary: #6366f1;
                --secondary: #a855f7;
                --bg: #0f172a;
                --card: rgba(30, 41, 59, 0.7);
            }}
            body {{
                background-color: var(--bg);
                color: white;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                margin: 0;
                padding: 40px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .container {{
                max-width: 1100px;
                width: 100%;
            }}
            h1 {{
                font-size: 2.5rem;
                margin-bottom: 10px;
                background: linear-gradient(to right, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .status {{
                color: #94a3b8;
                margin-bottom: 40px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 30px;
            }}
            .card {{
                background: var(--card);
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                font-size: 0.9rem;
            }}
            .data-table th {{
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid rgba(255,255,255,0.2);
                color: var(--primary);
            }}
            .data-table td {{
                padding: 12px;
                border-bottom: 1px solid rgba(255,255,255,0.05);
            }}
            .data-table tr:hover {{
                background: rgba(255,255,255,0.02);
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>☄️ Boom: Trajectory Unknown</h1>
            <p class="status">AI Generated Scenarios for Optimized Debris patterns</p>
            
            <div class="grid">
                <div class="card">
                    <h2>Impact Energy vs Debris Radius (P80/R95)</h2>
                    <canvas id="resultChart"></canvas>
                </div>

                <div class="card">
                    <h2>Top 20 Valid Scenarios</h2>
                    <div style="overflow-x: auto;">
                        {table_html}
                    </div>
                </div>
            </div>
        </div>

        <script>
            const ctx = document.getElementById('resultChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [
                        {{
                            label: 'P80 (Interaction Power)',
                            data: {json.dumps(p80_data)},
                            borderColor: '#6366f1',
                            backgroundColor: 'rgba(99, 102, 241, 0.2)',
                            tension: 0.4,
                            fill: true
                        }},
                        {{
                            label: 'R95 (Dispersal Area)',
                            data: {json.dumps(r95_data)},
                            borderColor: '#a855f7',
                            backgroundColor: 'rgba(168, 85, 247, 0.2)',
                            tension: 0.4,
                            fill: true
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{ labels: {{ color: 'white' }} }}
                    }},
                    scales: {{
                        y: {{ grid: {{ color: 'rgba(255,255,255,0.1)' }}, ticks: {{ color: 'white' }} }},
                        x: {{ grid: {{ display: false }}, ticks: {{ color: 'white' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Preview Dashboard generated at {output_path}")

if __name__ == "__main__":
    generate_html_dashboard()
