import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class DeploymentSimulator:
    """
    Simulate deployment scenarios and predict outcomes.
    
    Uses trained attrition and readiness models to:
    - Filter eligible personnel
    - Predict attrition during deployment
    - Calculate deployment readiness
    - Generate alternative scenarios
    """
    
    def __init__(self, data_path, attrition_model_path, readiness_model_path, 
                 attrition_scaler_path=None, readiness_scaler_path=None):
        """
        Initialize simulator with data and models.
        
        Parameters:
        -----------
        data_path : str
            Path to personnel dataset CSV
        attrition_model_path : str
            Path to trained attrition classifier
        readiness_model_path : str
            Path to trained readiness regressor
        attrition_scaler_path : str, optional
            Path to attrition feature scaler
        readiness_scaler_path : str, optional
            Path to readiness feature scaler
        """
        # Load data
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} personnel records")
        
        # Load models
        self.attrition_model = joblib.load(attrition_model_path)
        self.readiness_model = joblib.load(readiness_model_path)
        print("Models loaded")
        
        # Load scalers if provided
        self.attrition_scaler = joblib.load(attrition_scaler_path) if attrition_scaler_path else None
        self.readiness_scaler = joblib.load(readiness_scaler_path) if readiness_scaler_path else None
        
        # Prepare features
        self._prepare_features()
        
    def _prepare_features(self):
        """Prepare feature matrices for predictions."""
        # For ATTRITION model: keep unit_avg_readiness and relative_readiness
        # For READINESS model: remove them (leakage features)
        
        attrition_exclude = ['attrition_risk', 'readiness_score', 'personnel_id', 'name']
        readiness_exclude = ['attrition_risk', 'readiness_score', 'personnel_id', 'name',
                            'unit_avg_readiness', 'relative_readiness']
        
        # Prepare for attrition model (keeps all features)
        attrition_features = [col for col in self.df.columns if col not in attrition_exclude]
        self.df_attrition = pd.get_dummies(self.df[attrition_features], drop_first=True)
        
        # Prepare for readiness model (removes leakage)
        readiness_features = [col for col in self.df.columns if col not in readiness_exclude]
        self.df_readiness = pd.get_dummies(self.df[readiness_features], drop_first=True)
        
        print(f"Attrition features: {len(self.df_attrition.columns)}")
        print(f"Readiness features: {len(self.df_readiness.columns)}")
        
    def predict_attrition_risk(self):
        """Predict attrition risk for all personnel."""
        X = self.df_attrition
        
        if self.attrition_scaler:
            X = self.attrition_scaler.transform(X)
        
        predictions = self.attrition_model.predict(X)
        probabilities = self.attrition_model.predict_proba(X)
        
        # Map predictions back to labels
        label_map = {0: 'HIGH_RISK', 1: 'LOW_RISK', 2: 'MEDIUM_RISK'}
        self.df['predicted_attrition_risk'] = [label_map[p] for p in predictions]
        self.df['attrition_probability'] = probabilities[:, 0]  # HIGH_RISK probability
        
        return self.df[['predicted_attrition_risk', 'attrition_probability']]
    
    def predict_readiness_score(self):
        """Predict readiness scores for all personnel."""
        X = self.df_readiness
        
        if self.readiness_scaler:
            X = self.readiness_scaler.transform(X)
        
        predictions = self.readiness_model.predict(X)
        self.df['predicted_readiness_score'] = predictions
        
        return self.df['predicted_readiness_score']
    
    def simulate_deployment(self, n_personnel, duration_months=6, 
                           min_readiness=80, mos_requirements=None,
                           allow_high_risk=False, strategy='readiness'):
        """
        Simulate a deployment scenario.
        
        Parameters:
        -----------
        n_personnel : int
            Number of personnel needed
        duration_months : int
            Deployment duration in months
        min_readiness : float
            Minimum readiness score required (0-100)
        mos_requirements : list, optional
            Required MOS specialties
        allow_high_risk : bool
            Allow high attrition risk personnel
        strategy : str
            Selection strategy: 'readiness', 'balanced', 'low_risk'
        
        Returns:
        --------
        dict : Simulation results
        """
        print(f"\n{'-'*80}")
        print(f"DEPLOYMENT SIMULATION")
        print(f"{'-'*80}")
        print(f"Personnel needed: {n_personnel}")
        print(f"Duration: {duration_months} months")
        print(f"Min readiness: {min_readiness}")
        print(f"Strategy: {strategy}")
        
        # Ensure predictions are available
        if 'predicted_attrition_risk' not in self.df.columns:
            self.predict_attrition_risk()
        if 'predicted_readiness_score' not in self.df.columns:
            self.predict_readiness_score()
        
        # Filter eligible personnel
        eligible = self.df.copy()
        
        # Readiness filter
        eligible = eligible[eligible['predicted_readiness_score'] >= min_readiness]
        print(f"\nAfter readiness filter (>={min_readiness}): {len(eligible)} eligible")
        
        # Risk filter
        if not allow_high_risk:
            eligible = eligible[eligible['predicted_attrition_risk'] != 'HIGH_RISK']
            print(f"After risk filter (no HIGH_RISK): {len(eligible)} eligible")
        
        # MOS filter
        if mos_requirements:
            eligible = eligible[eligible['MOS'].isin(mos_requirements)]
            print(f"After MOS filter {mos_requirements}: {len(eligible)} eligible")
        
        # Check if enough personnel
        if len(eligible) < n_personnel:
            print(f"\nWARNING: Only {len(eligible)} eligible (need {n_personnel})")
            print("Consider lowering requirements or see alternative scenarios below.")
        
        # Select personnel based on strategy
        if strategy == 'readiness':
            # Highest readiness first
            selected = eligible.nlargest(min(n_personnel, len(eligible)), 'predicted_readiness_score')
        elif strategy == 'low_risk':
            # Lowest attrition probability first
            selected = eligible.nsmallest(min(n_personnel, len(eligible)), 'attrition_probability')
        else:  # balanced
            # Composite score: readiness - attrition_probability*10
            eligible['composite_score'] = (eligible['predicted_readiness_score'] - 
                                          eligible['attrition_probability'] * 10)
            selected = eligible.nlargest(min(n_personnel, len(eligible)), 'composite_score')
        
        # Calculate outcomes
        results = self._calculate_deployment_outcomes(selected, duration_months)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(n_personnel, duration_months, 
                                                   min_readiness, allow_high_risk)
        results['alternatives'] = alternatives
        
        return results
    
    def _calculate_deployment_outcomes(self, selected, duration_months):
        """Calculate predicted outcomes for selected personnel."""
        n_selected = len(selected)
        
        if n_selected == 0:
            return {
                'n_selected': 0,
                'avg_readiness': 0,
                'avg_attrition_probability': 0,
                'risk_distribution': {},
                'expected_attrition_count': 0,
                'expected_attrition_rate': 0,
                'replacement_needed': 0,
                'selected_personnel': []
            }
        
        # Average metrics
        avg_readiness = selected['predicted_readiness_score'].mean()
        avg_attrition_prob = selected['attrition_probability'].mean()
        
        # Risk distribution
        risk_dist = selected['predicted_attrition_risk'].value_counts()
        
        # Predict attrition during deployment
        base_attrition_rate = avg_attrition_prob * (duration_months / 12)
        expected_attrition_count = int(n_selected * base_attrition_rate)
        
        # Replacement needs
        replacement_needed = expected_attrition_count
        
        results = {
            'n_selected': n_selected,
            'avg_readiness': round(avg_readiness, 2),
            'avg_attrition_probability': round(avg_attrition_prob, 3),
            'risk_distribution': risk_dist.to_dict(),
            'expected_attrition_count': expected_attrition_count,
            'expected_attrition_rate': round(base_attrition_rate * 100, 1),
            'replacement_needed': replacement_needed,
            'selected_personnel': selected[['age', 'rank', 'MOS', 'years_of_service',
                                           'predicted_readiness_score', 
                                           'predicted_attrition_risk']].to_dict('records')
        }
        
        print(f"\n{'-'*80}")
        print(f"DEPLOYMENT OUTCOMES")
        print(f"{'-'*80}")
        print(f"Selected: {n_selected} personnel")
        print(f"Average readiness: {results['avg_readiness']:.1f}")
        print(f"Risk distribution: {risk_dist.to_dict()}")
        print(f"Expected attrition: {expected_attrition_count} personnel ({results['expected_attrition_rate']:.1f}%)")
        print(f"Replacement needed: {replacement_needed}")
        
        return results
    
    def _generate_alternatives(self, n_personnel, duration_months, 
                              min_readiness, allow_high_risk):
        """Generate alternative scenarios."""
        alternatives = []
        
        # Alternative 1: Lower readiness threshold
        if min_readiness > 70:
            alt_readiness = min_readiness - 5
            alt_eligible = self.df[self.df['predicted_readiness_score'] >= alt_readiness]
            if not allow_high_risk:
                alt_eligible = alt_eligible[alt_eligible['predicted_attrition_risk'] != 'HIGH_RISK']
            
            alternatives.append({
                'scenario': f'Lower readiness to {alt_readiness}',
                'eligible_pool': len(alt_eligible),
                'avg_readiness': round(alt_eligible['predicted_readiness_score'].mean(), 1),
                'trade_off': f'{len(alt_eligible)} total personnel available'
            })
        
        # Alternative 2: Allow high-risk personnel
        if not allow_high_risk:
            alt_eligible = self.df[self.df['predicted_readiness_score'] >= min_readiness]
            
            alternatives.append({
                'scenario': 'Allow HIGH_RISK personnel',
                'eligible_pool': len(alt_eligible),
                'high_risk_count': len(alt_eligible[alt_eligible['predicted_attrition_risk'] == 'HIGH_RISK']),
                'trade_off': 'Higher attrition risk'
            })
        
        # Alternative 3: Recruit/train more
        current_eligible = len(self.df[
            (self.df['predicted_readiness_score'] >= min_readiness) &
            (self.df['predicted_attrition_risk'] != 'HIGH_RISK' if not allow_high_risk else True)
        ])
        shortfall = max(0, n_personnel - current_eligible)
        if shortfall > 0:
            alternatives.append({
                'scenario': 'Recruit/train additional personnel',
                'personnel_needed': shortfall,
                'estimated_time': f'{shortfall * 6} months (6 months/person)',
                'trade_off': 'Time delay'
            })
        
        print(f"\n{'-'*80}")
        print(f"ALTERNATIVE SCENARIOS")
        print(f"{'-'*80}")
        for i, alt in enumerate(alternatives, 1):
            print(f"\n{i}. {alt['scenario']}")
            for key, value in alt.items():
                if key != 'scenario':
                    print(f"   {key}: {value}")
        
        return alternatives
    
    def compare_strategies(self, n_personnel, duration_months=6, min_readiness=80):
        """Compare different selection strategies."""
        strategies = ['readiness', 'balanced', 'low_risk']
        results = {}
        
        print(f"\n{'-'*80}")
        print(f"STRATEGY COMPARISON")
        print(f"{'-'*80}")
        
        for strategy in strategies:
            result = self.simulate_deployment(
                n_personnel=n_personnel,
                duration_months=duration_months,
                min_readiness=min_readiness,
                strategy=strategy
            )
            results[strategy] = result
        
        # Summary comparison
        comparison_df = pd.DataFrame({
            'Strategy': strategies,
            'Avg Readiness': [results[s]['avg_readiness'] for s in strategies],
            'Expected Attrition (%)': [results[s]['expected_attrition_rate'] for s in strategies],
            'Replacements Needed': [results[s]['replacement_needed'] for s in strategies]
        })
        
        print(f"\n{'-'*80}")
        print("COMPARISON SUMMARY")
        print(f"{'-'*80}")
        print(comparison_df.to_string(index=False))
        
        return results, comparison_df


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    import config
    
    # Initialize simulator
    simulator = DeploymentSimulator(
        data_path=config.PROCESSED_DATA_DIR / 'features_engineered.csv',
        attrition_model_path=config.MODELS_DIR / 'attrition_classifier.pkl',
        readiness_model_path=config.MODELS_DIR / 'readiness_regressor.pkl',
        readiness_scaler_path=config.MODELS_DIR / 'readiness_scaler.pkl'
    )
    
    # Run simulation
    results = simulator.simulate_deployment(
        n_personnel=50,
        duration_months=6,
        min_readiness=80,
        mos_requirements=['Infantry', 'Medical', 'Logistics'],
        allow_high_risk=False,
        strategy='balanced'
    )
    
    # Compare strategies
    comparison = simulator.compare_strategies(
        n_personnel=50,
        duration_months=6,
        min_readiness=80
    )