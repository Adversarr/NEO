import time
import csv
import os
import argparse
import psutil
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich import box

HISTORY_WINDOWS = {
    "1h": timedelta(hours=1),
    "3h": timedelta(hours=3),
    "24h": timedelta(hours=24),
    "3d": timedelta(days=3)
}

class DeviceMonitor:
    def __init__(self, csv_path: str, interval_s: float, top_users: int, sort_key: str):
        self.console = Console()
        self.data_file = csv_path
        self.interval_s = interval_s
        self.top_users = top_users
        self.sort_key = sort_key
        self.ensure_csv_header()
        self.gpu_mapping = {}
        self.gpu_details = {}
        self.history_df = pd.DataFrame()
        self.load_history()
        self.last_cpu_percent = psutil.cpu_percent(interval=None)
        self.boot_time = datetime.fromtimestamp(psutil.boot_time())
        # For network/disk IO speed
        self.last_net_io = psutil.net_io_counters()
        self.last_disk_io = psutil.disk_io_counters()
        self.last_io_time = time.time()

    def get_color_style(self, value, thresholds=(50, 85)):
        """Helper to get color style based on value and thresholds."""
        if value >= thresholds[1]:
            return "bold red"
        if value >= thresholds[0]:
            return "bold yellow"
        return "green"

    def format_bytes(self, n):
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if n < 1024:
                return f"{n:.1f}{unit}"
            n /= 1024
        return f"{n:.1f}PB"

    def load_history(self):
        if os.path.exists(self.data_file):
            try:
                self.history_df = pd.read_csv(self.data_file)
                self.history_df['timestamp'] = pd.to_datetime(self.history_df['timestamp'])
                # Fill missing proc_count if loading old CSV
                if 'proc_count' not in self.history_df.columns:
                    self.history_df['proc_count'] = 0
                cutoff = datetime.now() - timedelta(days=3)
                self.history_df = self.history_df[self.history_df['timestamp'] >= cutoff]
            except:
                self.history_df = pd.DataFrame()

    def update_history(self, metrics):
        if not metrics: return
        new_df = pd.DataFrame(metrics)
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        if self.history_df.empty:
            self.history_df = new_df
        else:
            self.history_df = pd.concat([self.history_df, new_df], ignore_index=True)
        
        cutoff = datetime.now() - timedelta(days=3)
        self.history_df = self.history_df[self.history_df['timestamp'] >= cutoff]

    def ensure_csv_header(self):
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'username', 'cpu_percent', 'mem_percent', 'gpu_mem_used', 'gpu_util_max', 'gpu_indices', 'proc_count'])

    def _run_command_lines(self, cmd):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode('utf-8').strip()
        except Exception:
            return []
        if not out:
            return []
        return [line.strip() for line in out.split('\n') if line.strip()]

    def get_gpu_details(self):
        try:
            cmd = ['nvidia-smi', '--query-gpu=index,pci.bus_id,utilization.gpu,memory.total', '--format=csv,noheader,nounits']
            self.gpu_mapping = {}
            self.gpu_details = {}
            for line in self._run_command_lines(cmd):
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 4:
                    idx = int(parts[0])
                    bus_id = parts[1]
                    util = float(parts[2])
                    total_mem = float(parts[3])
                    self.gpu_mapping[bus_id] = idx
                    self.gpu_details[idx] = {'util': util, 'total_mem': total_mem, 'bus_id': bus_id}
        except Exception:
            pass

    def get_gpu_summary_rows(self):
        rows = []
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,fan.speed',
            '--format=csv,noheader,nounits'
        ]
        for line in self._run_command_lines(cmd):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 10:
                continue
            try:
                rows.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'temp_c': float(parts[2]),
                    'gpu_util': float(parts[3]),
                    'mem_util': float(parts[4]), # This is memory bus utilization
                    'mem_used': float(parts[5]),
                    'mem_total': float(parts[6]),
                    'power_draw': float(parts[7]),
                    'power_limit': float(parts[8]),
                    'fan_speed': parts[9], # Can be "[N/A]"
                })
            except Exception:
                continue
        return rows

    def get_gpu_processes(self):
        gpu_procs = {}
        cmd = ['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id,used_memory', '--format=csv,noheader,nounits']
        for line in self._run_command_lines(cmd):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                bus_id = parts[1]
                mem_used = float(parts[2])
            except Exception:
                continue
            gpu_idx = self.gpu_mapping.get(bus_id, -1)
            if gpu_idx != -1:
                gpu_procs[pid] = {'gpu_mem': mem_used, 'gpu_idx': gpu_idx}
        return gpu_procs

    def collect_metrics(self):
        self.get_gpu_details()
        gpu_procs = self.get_gpu_processes()

        user_metrics = defaultdict(lambda: {
            'cpu_percent': 0.0,
            'mem_percent': 0.0,
            'gpu_mem_used': 0.0,
            'gpu_indices': set(),
            'gpu_utils': [],
            'gpu_mem_by_idx': defaultdict(float),
            'proc_count': 0
        })

        if not hasattr(self, 'process_cache'):
            self.process_cache = {}

        current_pids = set()
        
        for p in psutil.process_iter(['pid', 'username', 'memory_percent', 'status']):
            try:
                pid = p.info['pid']
                current_pids.add(pid)
                
                if pid in self.process_cache:
                    proc_obj = self.process_cache[pid]
                else:
                    proc_obj = psutil.Process(pid)
                    proc_obj.cpu_percent(interval=None)
                    self.process_cache[pid] = proc_obj

                try:
                    cpu = proc_obj.cpu_percent(interval=None)
                except:
                    cpu = 0.0
                
                mem = p.info['memory_percent'] or 0.0
                username = p.info['username']

                if username:
                    user_metrics[username]['proc_count'] += 1
                    user_metrics[username]['cpu_percent'] += cpu
                    user_metrics[username]['mem_percent'] += mem
                    
                    if pid in gpu_procs:
                        g_info = gpu_procs[pid]
                        user_metrics[username]['gpu_mem_used'] += g_info['gpu_mem']
                        idx = g_info['gpu_idx']
                        user_metrics[username]['gpu_indices'].add(idx)
                        user_metrics[username]['gpu_mem_by_idx'][idx] += g_info['gpu_mem']
                        if idx in self.gpu_details:
                            user_metrics[username]['gpu_utils'].append(self.gpu_details[idx]['util'])

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        self.process_cache = {k: v for k, v in self.process_cache.items() if k in current_pids}

        results = []
        timestamp = datetime.now().isoformat()
        
        for user, data in user_metrics.items():
            if data['cpu_percent'] < 0.1 and data['mem_percent'] < 0.1 and data['gpu_mem_used'] == 0:
                continue

            max_gpu_util = max(data['gpu_utils']) if data['gpu_utils'] else 0.0
            gpu_indices_str = ";".join(map(str, sorted(list(data['gpu_indices']))))
            gpu_mem_by_gpu_str = " ".join(
                [f"{idx}:{int(mem)}MB" for idx, mem in sorted(data['gpu_mem_by_idx'].items(), key=lambda x: x[0])]
            )
            
            results.append({
                'timestamp': timestamp,
                'username': user,
                'cpu_percent': round(data['cpu_percent'], 2),
                'mem_percent': round(data['mem_percent'], 2),
                'gpu_mem_used': round(data['gpu_mem_used'], 2),
                'gpu_util_max': round(max_gpu_util, 2),
                'gpu_indices': gpu_indices_str,
                'gpu_mem_by_gpu': gpu_mem_by_gpu_str,
                'proc_count': data['proc_count']
            })
            
        return results

    def save_metrics(self, metrics):
        if not metrics:
            return
        
        with open(self.data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for m in metrics:
                writer.writerow([
                    m['timestamp'], m['username'], m['cpu_percent'], m['mem_percent'],
                    m['gpu_mem_used'], m['gpu_util_max'], m['gpu_indices'], m.get('proc_count', 0)
                ])

    def get_aggregated_stats(self):
        if self.history_df.empty:
            return {}

        try:
            df = self.history_df
            now = datetime.now()
            
            stats = {}
            
            for label, delta in HISTORY_WINDOWS.items():
                cutoff = now - delta
                mask = df['timestamp'] >= cutoff
                subset = df[mask]
                
                if subset.empty:
                    continue
                
                grouped = subset.groupby('username').agg({
                    'cpu_percent': 'mean',
                    'mem_percent': 'mean',
                    'gpu_mem_used': 'mean',
                    'gpu_util_max': 'mean',
                    'proc_count': 'mean'
                }).reset_index()

                period_dict = grouped.set_index('username').to_dict(orient='index')
                for u, v in period_dict.items():
                    cpu_v = float(v.get('cpu_percent', 0.0) or 0.0)
                    mem_v = float(v.get('mem_percent', 0.0) or 0.0)
                    gmem_v = float(v.get('gpu_mem_used', 0.0) or 0.0)
                    gutil_v = float(v.get('gpu_util_max', 0.0) or 0.0)
                    v['score'] = (cpu_v * 1.0) + (mem_v * 0.1) + (gutil_v * 1.0) + (gmem_v / 1024.0)
                stats[label] = period_dict
                
            return stats
        except Exception:
            return {}

    def _compute_top_users(self, history_stats):
        tops = {}
        for label, d in history_stats.items():
            ranked = sorted(d.items(), key=lambda kv: float(kv[1].get('score', 0.0) or 0.0), reverse=True)
            top1 = ranked[0][0] if len(ranked) >= 1 else None
            top2 = ranked[1][0] if len(ranked) >= 2 else None
            tops[label] = (top1, top2)
        return tops

    def render_gpu_table(self, gpu_rows):
        table = Table(title="🎮 GPU Summary", box=box.ROUNDED, show_lines=False)
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Name", style="white")
        table.add_column("Temp", justify="right")
        table.add_column("Fan", justify="right")
        table.add_column("GPU%", justify="right")
        table.add_column("Bus%", justify="right")
        table.add_column("Memory Usage", justify="right")
        table.add_column("Power", justify="right")

        for r in sorted(gpu_rows, key=lambda x: x.get('index', 0)):
            mem_str = f"{r['mem_used']:.0f}/{r['mem_total']:.0f}MB"
            power_str = f"{r['power_draw']:.0f}/{r['power_limit']:.0f}W"
            gpu_util = r.get('gpu_util', 0.0)
            mem_util = r.get('mem_util', 0.0)
            temp = r.get('temp_c', 0.0)
            
            # Dynamic coloring
            gpu_style = self.get_color_style(gpu_util)
            mem_style = self.get_color_style(mem_util)
            temp_style = self.get_color_style(temp, thresholds=(70, 85))
            
            table.add_row(
                str(r.get('index', '-')),
                r.get('name', '-')[:26],
                Text(f"{temp:.0f}°C", style=temp_style),
                f"{r.get('fan_speed', '-'):>3}%" if str(r.get('fan_speed')).isdigit() else r.get('fan_speed', '-'),
                Text(f"{gpu_util:.0f}%", style=gpu_style),
                Text(f"{mem_util:.0f}%", style=mem_style),
                mem_str,
                power_str,
            )
        return table

    def render_system_panel(self):
        cpu_now = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        
        # Uptime
        uptime = datetime.now() - self.boot_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        uptime_str = f"{days}d {hours}h {minutes}m"

        # Load
        try:
            la1, la5, la15 = os.getloadavg()
            load_str = f"{la1:.2f} {la5:.2f} {la15:.2f}"
        except Exception:
            load_str = "-"

        # Disk usage (root)
        du = psutil.disk_usage('/')
        
        # IO Speed
        now = time.time()
        dt = now - self.last_io_time
        
        net_now = psutil.net_io_counters()
        net_in = (net_now.bytes_recv - self.last_net_io.bytes_recv) / dt
        net_out = (net_now.bytes_sent - self.last_net_io.bytes_sent) / dt
        
        disk_now = psutil.disk_io_counters()
        disk_read = (disk_now.read_bytes - self.last_disk_io.read_bytes) / dt
        disk_write = (disk_now.write_bytes - self.last_disk_io.write_bytes) / dt
        
        self.last_net_io = net_now
        self.last_disk_io = disk_now
        self.last_io_time = now

        lines = [
            f"🧠 CPU: [bold cyan]{cpu_now:5.1f}%[/]  Load: {load_str}",
            f"🧠 RAM: [bold magenta]{vm.percent:5.1f}%[/]  {vm.used/1024**3:5.1f}/{vm.total/1024**3:5.1f} GB",
            f"💾 Disk: [bold yellow]{du.percent:5.1f}%[/]  {du.used/1024**3:5.1f}/{du.total/1024**3:5.1f} GB",
            f"🌐 Net:  ⬇️ {self.format_bytes(net_in)}/s  ⬆️ {self.format_bytes(net_out)}/s",
            f"💽 I/O:  📖 {self.format_bytes(disk_read)}/s  📝 {self.format_bytes(disk_write)}/s",
            f"⏱️ Uptime: {uptime_str}",
        ]
        return Panel(Text.from_markup("\n".join(lines)), title="💻 System Info", box=box.ROUNDED)

    def render_users_table(self, current_metrics, history_stats):
        tops = self._compute_top_users(history_stats)
        curr_dict = {m['username']: m for m in current_metrics}

        all_users = set(curr_dict.keys())
        for period in history_stats:
            all_users.update(history_stats[period].keys())

        def user_sort_key(u):
            if self.sort_key == "cpu":
                return float(curr_dict.get(u, {}).get('cpu_percent', 0.0) or 0.0)
            if self.sort_key == "gpu_mem":
                return float(curr_dict.get(u, {}).get('gpu_mem_used', 0.0) or 0.0)
            if self.sort_key == "score":
                h = history_stats.get("1h", {}).get(u, {})
                return float(h.get('score', 0.0) or 0.0)
            return u

        reverse = self.sort_key in {"cpu", "gpu_mem", "score"}
        users_sorted = sorted(list(all_users), key=user_sort_key, reverse=reverse)
        if self.top_users > 0:
            users_sorted = users_sorted[: self.top_users]

        table = Table(title="👥 User Resource Consumption", box=box.ROUNDED, show_lines=False)
        table.add_column("User", style="cyan", no_wrap=True)
        table.add_column("Current Real-time", justify="left")
        table.add_column("GPU Mem (by ID)", justify="left")
        for label in ["1h Avg", "3h Avg", "24h Avg", "3d Avg"]:
            table.add_column(label, justify="left")

        for u in users_sorted:
            cm = curr_dict.get(u)
            if cm:
                gpus = cm.get('gpu_indices') or "-"
                procs = cm.get('proc_count', 0)
                
                # Dynamic styles
                c_style = self.get_color_style(cm['cpu_percent'], thresholds=(100, 400)) # Multi-core
                m_style = self.get_color_style(cm['mem_percent'], thresholds=(10, 30))
                g_style = self.get_color_style(cm['gpu_util_max'])
                
                current_cell = Text.from_markup(
                    f"[bold]P:{procs}[/] | "
                    f"C:[{c_style}]{cm['cpu_percent']:>5.1f}%[/] | "
                    f"M:[{m_style}]{cm['mem_percent']:>4.1f}%[/] | "
                    f"G:[{g_style}]{cm['gpu_util_max']:>3.0f}%[/] | "
                    f"GMem:[bold]{cm['gpu_mem_used']:>5.0f}MB[/] | "
                    f"ID:{gpus}"
                )
                gpu_by_gpu_cell = cm.get('gpu_mem_by_gpu') or "-"
            else:
                current_cell = Text("-", style="dim")
                gpu_by_gpu_cell = Text("-", style="dim")

            period_cells = []
            for label in ["1h", "3h", "24h", "3d"]:
                if label in history_stats and u in history_stats[label]:
                    h = history_stats[label][u]
                    
                    c_style = self.get_color_style(h['cpu_percent'], thresholds=(100, 400))
                    g_style = self.get_color_style(h['gpu_util_max'])
                    
                    s = f"C:{h['cpu_percent']:>5.1f}% | G:{h['gpu_util_max']:>3.0f}% | GM:{h['gpu_mem_used']:>5.0f}M"
                    
                    top1, top2 = tops.get(label, (None, None))
                    if u == top1:
                        period_cells.append(Text(s, style="bold white on red"))
                    elif u == top2:
                        period_cells.append(Text(s, style="bold black on yellow"))
                    else:
                        period_cells.append(Text.from_markup(f"C:[{c_style}]{h['cpu_percent']:>5.1f}%[/] | G:[{g_style}]{h['gpu_util_max']:>3.0f}%[/] | GM:{h['gpu_mem_used']:>5.0f}M"))
                else:
                    period_cells.append(Text("-", style="dim"))

            table.add_row(u, current_cell, gpu_by_gpu_cell, *period_cells)

        return table

    def render_dashboard(self, current_metrics, history_stats, gpu_rows):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = Panel(
            Text(
                f"🚀 Device Monitor  |  🕒 {now_str}  |  🔄 {self.interval_s:.1f}s  |  📂 {os.path.abspath(self.data_file)}",
                justify="center",
                style="bold white on blue"
            ),
            box=box.DOUBLE_EDGE,
        )

        sys_panel = self.render_system_panel()
        gpu_table = self.render_gpu_table(gpu_rows)
        gpu_panel = Panel(gpu_table, box=box.ROUNDED)

        users_table = self.render_users_table(current_metrics, history_stats)
        users_panel = Panel(users_table, box=box.ROUNDED)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="upper", size=10),
            Layout(name="users", ratio=1),
        )
        layout["header"].update(header)
        layout["upper"].split_row(
            Layout(name="system", ratio=1),
            Layout(name="gpus", ratio=2),
        )
        layout["upper"]["system"].update(sys_panel)
        layout["upper"]["gpus"].update(gpu_panel)
        layout["users"].update(users_panel)
        return layout

    def run(self):
        with Live(console=self.console, refresh_per_second=1, screen=True) as live:
            while True:
                try:
                    metrics = self.collect_metrics()
                    self.save_metrics(metrics)
                    self.update_history(metrics)
                    
                    stats = self.get_aggregated_stats()
                    
                    gpu_rows = self.get_gpu_summary_rows()
                    dashboard = self.render_dashboard(metrics, stats, gpu_rows)
                    live.update(dashboard)
                    
                    time.sleep(self.interval_s)
                except KeyboardInterrupt:
                    break
                except Exception:
                    time.sleep(self.interval_s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="device_monitor.py")
    parser.add_argument("--csv", default="device_monitor_data.csv")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--top-users", type=int, default=40)
    parser.add_argument("--sort", choices=["cpu", "gpu_mem", "score", "name"], default="score")
    args = parser.parse_args()
    sort_key = args.sort
    if sort_key == "name":
        sort_key = "name"
    monitor = DeviceMonitor(csv_path=args.csv, interval_s=args.interval, top_users=args.top_users, sort_key=sort_key)
    monitor.run()
