"""HTML Dashboard Generator for Validation Laboratory."""

import os
from typing import Dict, Any, List

class DashboardGenerator:
    """Generates the standalone Validation Laboratory HTML Dashboard."""

    @staticmethod
    def generate(results: List[Dict[str, Any]], metrics: Dict[str, Any], output_path: str):
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ModelDoctor Validation Laboratory</title>
    <style>
        :root {{
            --primary: #007aff;
            --success: #34c759;
            --warning: #ffcc00;
            --error: #ff3b30;
            --bg: #f5f5f7;
            --card-bg: #ffffff;
            --text-main: #1d1d1f;
            --text-muted: #86868b;
            --border: #d2d2d7;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{
            font-family: var(--font-family);
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 2rem;
            -webkit-font-smoothing: antialiased;
        }}
        h1, h2, h3 {{ font-weight: 600; letter-spacing: -0.015em; }}
        h1 {{ font-size: 2.5rem; text-align: center; margin-bottom: 2rem; }}
        
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
        
        .card {{
            background: var(--card-bg);
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }}
        
        .stat-value {{ font-size: 3rem; font-weight: 700; color: var(--primary); }}
        .stat-label {{ font-size: 0.9rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }}
        
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        
        .badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .badge.pass {{ background: rgba(52, 199, 89, 0.15); color: var(--success); }}
        .badge.fail {{ background: rgba(255, 59, 48, 0.15); color: var(--error); }}
        
        .progress-bar-bg {{ background: var(--bg); height: 8px; border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }}
        .progress-bar-fill {{ background: var(--primary); height: 100%; }}
        
        .doctor-list {{ list-style: none; padding: 0; margin: 0; }}
        .doctor-item {{ display: flex; justify-content: space-between; padding: 0.75rem 0; border-bottom: 1px solid var(--border); }}
        .doctor-item:last-child {{ border-bottom: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Laboratory</h1>
        
        <div class="grid">
            <div class="card">
                <div class="stat-label">Total Scenarios</div>
                <div class="stat-value">{metrics.get('total_scenarios', 0)}</div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: 100%;"></div>
                </div>
            </div>
            
            <div class="card">
                <div class="stat-label">Passed</div>
                <div class="stat-value" style="color: var(--success)">{metrics.get('passed', 0)}</div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: {metrics.get('overall_accuracy', 0)*100}%; background: var(--success)"></div>
                </div>
            </div>
            
            <div class="card">
                <div class="stat-label">Release Readiness</div>
                <div class="stat-value" style="color: {'var(--success)' if metrics.get('failed', 0) == 0 else 'var(--error)'}; font-size: 2rem; margin-top: 1rem;">
                    {'READY' if metrics.get('failed', 0) == 0 else 'NOT READY'}
                </div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Doctor Leaderboard</h2>
                <ul class="doctor-list">
"""
        for doctor, acc in sorted(metrics.get("per_doctor_accuracy", {}).items(), key=lambda x: x[1], reverse=True):
            html += f"""
                    <li class="doctor-item">
                        <span>{doctor.capitalize()}</span>
                        <span style="font-weight: 600; color: {'var(--success)' if acc == 1.0 else 'var(--error)'}">{acc*100:.0f}%</span>
                    </li>
            """
            
        html += f"""
                </ul>
            </div>
            
            <div class="card">
                <h2>Metrics Overview</h2>
                <ul class="doctor-list">
                    <li class="doctor-item">
                        <span>False Positives</span>
                        <span style="font-weight: 600">{metrics.get('false_positive_rate', 0)*100:.1f}%</span>
                    </li>
                    <li class="doctor-item">
                        <span>False Negatives</span>
                        <span style="font-weight: 600">{metrics.get('false_negative_rate', 0)*100:.1f}%</span>
                    </li>
                    <li class="doctor-item">
                        <span>Average Confidence</span>
                        <span style="font-weight: 600">{metrics.get('average_confidence', 0):.1f}%</span>
                    </li>
                    <li class="doctor-item">
                        <span>Average Runtime</span>
                        <span style="font-weight: 600">{metrics.get('average_runtime_s', 0):.2f}s</span>
                    </li>
                </ul>
            </div>
        </div>
        
        <div class="card">
            <h2>Pass/Fail Matrix</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Scenario</th>
                        <th>Status</th>
                        <th>Runtime</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
"""
        for r in results:
            status_class = "pass" if r.get("status") == "PASS" else "fail"
            html += f"""
                    <tr>
                        <td style="font-weight: 500">{r.get('category')}</td>
                        <td>{r.get('scenario')}</td>
                        <td><span class="badge {status_class}">{r.get('status')}</span></td>
                        <td>{r.get('duration_s', 0):.2f}s</td>
                        <td style="color: var(--error); font-size: 0.85rem;">{r.get('error') or ''}</td>
                    </tr>
            """
            
        html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
