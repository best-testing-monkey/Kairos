from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig, print_results

if __name__ == "__main__":
	config = OrchestratorConfig(
		initial_capital=10000.0,
		cross_asset_ranking=True,
		online_weighting=True,
		partial_exits=True,
		max_horizon=3,
	)

	orchestrator = KairosOrchestrator(
		predict_fn=predict_kairos_cloud,
		assets=["BTC-USD", "ETH-USD", "SOL-USD"],
		config=config,
	)

	results = orchestrator.run_backtest({
		"BTC-USD": btc_df,
		"ETH-USD": eth_df,
		"SOL-USD": sol_df,
	}, lookback=200)

	print_results(results)
