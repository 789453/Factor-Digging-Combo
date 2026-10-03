import duckdb
conn = duckdb.connect('D:/Trading/data_ever_26_3_14/data/meta/warehouse.duckdb', read_only=True)

print('=== 所有表 ===')
rows = conn.execute("SELECT table_schema || '.' || table_name as name FROM information_schema.tables WHERE table_schema NOT IN ('system', 'information_schema')").fetchall()
for r in rows:
    print(r[0])

print()
print('=== silver schema 表结构 ===')
for table in ['fact_stock_daily', 'fact_stock_moneyflow', 'fact_stock_cyq_perf', 'fact_stock_daily_basic', 'fact_stock_basic_snapshot']:
    try:
        cols = conn.execute(f"SELECT column_name FROM information_schema.columns WHERE table_schema='silver' AND table_name='{table}'").fetchall()
        print(f'{table}: {[c[0] for c in cols]}')
    except Exception as e:
        print(f'{table}: ERROR - {e}')

conn.close()