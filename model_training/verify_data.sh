# verify_data.sh
#!/usr/bin/env bash

echo "=========================================="
echo "DATA VERIFICATION SCRIPT"
echo "=========================================="

# Check if file exists
if [ ! -f "./data_stress_strain_differential_random_then_fixed_labeled.npz" ]; then
    echo "ERROR: Data file not found!"
    echo "Looking for: ./data_stress_strain_differential_random_then_fixed_labeled.npz"
    exit 1
fi

echo "Loading data..."

# Run Python verification
python3 - << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
import os

# Load data
try:
    data = np.load('./data_stress_strain_differential_random_then_fixed_labeled.npz')
    print("✅ Successfully loaded data")
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    exit(1)

# Check what's in the file
print("\n=== FILE CONTENTS ===")
print("Keys:", list(data.keys()))

required_keys = ['eps', 'epse', 'deps', 'sig', 'split']
for key in required_keys:
    if key in data:
        print(f"  ✅ {key}: shape = {data[key].shape}")
    else:
        print(f"  ❌ {key}: MISSING!")

# Basic statistics
print("\n=== BASIC STATISTICS ===")
for key in ['eps', 'epse', 'deps', 'sig']:
    if key in data:
        arr = data[key]
        print(f"\n{key}:")
        print(f"  Shape: {arr.shape}")
        print(f"  Global mean: {arr.mean():.6f}")
        print(f"  Global std:  {arr.std():.6f}")
        print(f"  Global min:  {arr.min():.6f}")
        print(f"  Global max:  {arr.max():.6f}")
        print(f"  NaN count:   {np.isnan(arr).sum()}")
        print(f"  Inf count:   {np.isinf(arr).sum()}")

# Check strain variability (CRITICAL!)
print("\n=== STRAIN VARIABILITY ANALYSIS ===")
eps = data['eps']
sig = data['sig']

print(f"Number of specimens: {eps.shape[0]}")
print(f"Time steps per specimen: {eps.shape[1]}")

# Check first 3 specimens in detail
print("\n--- Detailed analysis of first 3 specimens ---")
for spec_idx in range(min(3, eps.shape[0])):
    print(f"\n📊 Specimen {spec_idx}:")
    
    for comp in range(6):
        eps_comp = eps[spec_idx, :, comp]
        sig_comp = sig[spec_idx, :, comp]
        
        eps_min = eps_comp.min()
        eps_max = eps_comp.max()
        eps_std = eps_comp.std()
        eps_range = eps_max - eps_min
        
        sig_min = sig_comp.min()
        sig_max = sig_comp.max()
        sig_std = sig_comp.std()
        
        # Check if strain is constant
        is_constant = eps_std < 1e-6
        
        print(f"  Component {comp}:")
        print(f"    ε: [{eps_min:.6f}, {eps_max:.6f}]")
        print(f"    ε std: {eps_std:.6f}")
        print(f"    ε range: {eps_range:.6f}")
        
        if is_constant:
            print(f"    ⚠️  STRAIN IS CONSTANT! (std < 1e-6)")
        else:
            # Compute correlation
            if eps_std > 1e-6:  # Avoid division by zero
                correlation = np.corrcoef(eps_comp, sig_comp)[0, 1]
                print(f"    Correlation ε-σ: {correlation:.3f}")
            
        print(f"    σ: [{sig_min:.3f}, {sig_max:.3f}]")
        print(f"    σ std: {sig_std:.3f}")
        print()

# Check if ALL specimens have constant strain
print("\n=== CONSTANT STRAIN DETECTION ===")
constant_count = 0
total_components = eps.shape[0] * 6

for spec_idx in range(eps.shape[0]):
    for comp in range(6):
        if eps[spec_idx, :, comp].std() < 1e-6:
            constant_count += 1

print(f"Constant strain components: {constant_count}/{total_components} ({constant_count/total_components*100:.1f}%)")

if constant_count == total_components:
    print("❌ CRITICAL: ALL strain components are constant!")
    print("   Your model cannot learn anything from constant inputs.")
elif constant_count > 0:
    print("⚠️  WARNING: Some strain components are constant")
else:
    print("✅ Good: No constant strain components found")

# Check split distribution
print("\n=== DATA SPLIT DISTRIBUTION ===")
if 'split' in data:
    split = data['split']
    unique, counts = np.unique(split, return_counts=True)
    for s, c in zip(unique, counts):
        print(f"  {s}: {c} specimens")
else:
    print("  No split information found")

# Create visualization plots
print("\n=== CREATING VISUALIZATION PLOTS ===")
os.makedirs("data_verification_plots", exist_ok=True)

# Plot 1: Strain and stress over time for first specimen
fig1, axes1 = plt.subplots(3, 2, figsize=(12, 10))
fig1.suptitle('Strain and Stress vs Time (Specimen 0)', fontsize=14)

component_names = ['ε₁₁/σ₁₁', 'ε₂₂/σ₂₂', 'ε₃₃/σ₃₃', 'ε₂₃/σ₂₃', 'ε₁₃/σ₁₃', 'ε₁₂/σ₁₂']
time = np.arange(eps.shape[1])

for idx, (ax, name) in enumerate(zip(axes1.ravel(), component_names)):
    eps_comp = eps[0, :, idx]
    sig_comp = sig[0, :, idx]
    
    # Plot strain (left axis)
    color1 = 'tab:blue'
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Strain', color=color1)
    ax.plot(time, eps_comp, color=color1, linewidth=2, label='Strain')
    ax.tick_params(axis='y', labelcolor=color1)
    
    # Plot stress (right axis)
    ax2 = ax.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Stress', color=color2)
    ax2.plot(time, sig_comp, color=color2, linewidth=2, linestyle='--', label='Stress')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_verification_plots/strain_stress_time.png', dpi=150, bbox_inches='tight')
print("✅ Saved: data_verification_plots/strain_stress_time.png")

# Plot 2: Strain-stress relationships
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle('Stress vs Strain Relationships (Specimen 0)', fontsize=14)

pairs = [(0, 'σ₁₁ vs ε₁₁'), (1, 'σ₂₂ vs ε₂₂'), (2, 'σ₃₃ vs ε₃₃'),
         (3, 'σ₂₃ vs ε₂₃'), (4, 'σ₁₃ vs ε₁₃'), (5, 'σ₁₂ vs ε₁₂')]

for idx, (ax, (comp_idx, title)) in enumerate(zip(axes2.ravel(), pairs)):
    eps_comp = eps[0, :, comp_idx]
    sig_comp = sig[0, :, comp_idx]
    
    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(eps_comp)))
    
    # Plot with time coloring
    for i in range(len(eps_comp)-1):
        ax.plot([eps_comp[i], eps_comp[i+1]],
                [sig_comp[i], sig_comp[i+1]],
                color=colors[i], alpha=0.7, linewidth=2)
    
    # Mark start and end
    ax.scatter(eps_comp[0], sig_comp[0], color='green', s=100, 
               label='Start', marker='o', edgecolors='black', zorder=5)
    ax.scatter(eps_comp[-1], sig_comp[-1], color='red', s=100,
               label='End', marker='s', edgecolors='black', zorder=5)
    
    ax.set_xlabel(f'ε{title[1:4]}')
    ax.set_ylabel(title[:3])
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    
    # Add correlation if not constant
    if np.std(eps_comp) > 1e-6:
        corr = np.corrcoef(eps_comp, sig_comp)[0, 1]
        ax.text(0.05, 0.95, f'corr = {corr:.3f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('data_verification_plots/stress_vs_strain.png', dpi=150, bbox_inches='tight')
print("✅ Saved: data_verification_plots/stress_vs_strain.png")

# Plot 3: Check multiple specimens
fig3, axes3 = plt.subplots(3, 2, figsize=(12, 10))
fig3.suptitle('Strain Component 0 (ε₁₁) Across Different Specimens', fontsize=14)

# Plot ε₁₁ for first 6 specimens
for i in range(6):
    ax = axes3[i//2, i%2]
    eps_comp = eps[i, :, 0]
    sig_comp = sig[i, :, 0]
    
    ax.plot(time, eps_comp, 'b-', label=f'ε₁₁', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(time, sig_comp, 'r--', label=f'σ₁₁', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('ε₁₁', color='b')
    ax2.set_ylabel('σ₁₁', color='r')
    ax.set_title(f'Specimen {i}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_verification_plots/multiple_specimens.png', dpi=150, bbox_inches='tight')
print("✅ Saved: data_verification_plots/multiple_specimens.png")

# Summary statistics
print("\n=== SUMMARY ===")
print("1. Check the plots in 'data_verification_plots/' folder")
print("2. Look for:")
print("   - Strain varying over time (should see curves, not flat lines)")
print("   - Stress responding to strain changes")
print("   - Reasonable strain values (typically 10^-3 to 10^-2)")
print("   - Reasonable stress values (depends on material)")
print("\n3. Common problems:")
print("   - Constant strain: Model can't learn")
print("   - Constant stress: No output variation")
print("   - Strain and stress not correlated: Wrong physics")
print("   - NaN/Inf values: Data corruption")

plt.close('all')
print("\n✅ Data verification complete!")
EOF

echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="
echo "Check the 'data_verification_plots/' folder for visualizations."
echo "Look at the printed statistics above."