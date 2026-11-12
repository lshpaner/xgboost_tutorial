// Palette read from styles.css so plots follow the light / dark theme
function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function pal() {
    return {
        ink: cssVar('--ink'), muted: cssVar('--muted'), rule: cssVar('--rule'),
        panel: cssVar('--panel'), accent: cssVar('--accent'), moss: cssVar('--moss'),
        mossSoft: cssVar('--moss-soft'), grid: cssVar('--grid'),
        leaves: [cssVar('--leaf-1'), cssVar('--leaf-2'), cssVar('--leaf-3')]
    };
}
const FONT = '"Times New Roman", Times, serif';
const MONO = '"Times New Roman", Times, serif';
function axis(title) {
    const p = pal();
    return {
        gridcolor: p.grid, zerolinecolor: p.rule, linecolor: p.rule,
        tickfont: {size: 11, family: MONO, color: p.muted},
        title: {text: title, font: {size: 12, family: MONO, color: p.muted}}
    };
}
function baseLayout(titleText) {
    const p = pal();
    return {
        title: {text: titleText, x: 0, xanchor: 'left', font: {size: 13, family: FONT, color: p.ink}},
        paper_bgcolor: p.panel, plot_bgcolor: p.panel,
        font: {family: FONT, color: p.ink}
    };
}

        // Global variables for tree visualization
        let globalTrees = [];
        let globalX = [];
        let globalY = [];
        let globalPredictionHistory = [];
        let currentStepIndex = 0;
        let globalYMin = 0;
        let globalYMax = 0;
        let mainPlotYMin = 0;
        let mainPlotYMax = 0;
        let lastRun = null;

        function drawMainPlots() {
            if (!lastRun) return;
            const r = lastRun;
            plotPredictions(r.X, r.y, r.predictionHistory);
            plotResiduals(r.y, r.predictions);
            plotLearningCurve(r.mseHistory);
            displayMetrics(r.y, r.predictions, r.nEstimators);
        }

        function redrawForTheme() {
            if (!lastRun) return;
            drawMainPlots();
            visualizeTrees(globalTrees);
            updateSequentialView();
        }

        // Update slider values display
        document.getElementById('n-estimators').addEventListener('input', function(e) {
            document.getElementById('n-estimators-value').textContent = e.target.value;
        });
        
        document.getElementById('learning-rate').addEventListener('input', function(e) {
            document.getElementById('learning-rate-value').textContent = parseFloat(e.target.value).toFixed(2);
        });
        
        document.getElementById('max-depth').addEventListener('input', function(e) {
            document.getElementById('max-depth-value').textContent = e.target.value;
        });

        // Simple XGBoost simulation for regression
        function runXGBoostDemo() {
            const nEstimators = parseInt(document.getElementById('n-estimators').value);
            const learningRate = parseFloat(document.getElementById('learning-rate').value);
            const maxDepth = parseInt(document.getElementById('max-depth').value);

            // Generate synthetic data
            const nSamples = 100;
            const X = Array.from({length: nSamples}, (_, i) => (i / nSamples) * 10 - 5);
            const y = X.map(x => Math.sin(x) * x + (Math.random() - 0.5) * 2);

            // Store globally for tree visualization
            globalX = X;
            globalY = y;
            globalTrees = [];
            globalPredictionHistory = [];

            // Initialize predictions with mean
            let predictions = new Array(nSamples).fill(y.reduce((a, b) => a + b) / nSamples);
            const predictionHistory = [predictions.slice()];
            const mseHistory = [calculateMSE(y, predictions)];

            globalPredictionHistory.push({
                predictions: predictions.slice(),
                tree: null,
                mse: mseHistory[0]
            });

            // Boosting iterations
            for (let m = 0; m < nEstimators; m++) {
                const residuals = y.map((yi, i) => yi - predictions[i]);
                
                // Build tree and store structure
                const treeResult = fitSimpleTreeWithStructure(X, residuals, maxDepth, m);
                globalTrees.push(treeResult.tree);
                
                // Update predictions
                predictions = predictions.map((pred, i) => pred + learningRate * treeResult.predictions[i]);
                predictionHistory.push(predictions.slice());
                mseHistory.push(calculateMSE(y, predictions));

                globalPredictionHistory.push({
                    predictions: predictions.slice(),
                    tree: treeResult.tree,
                    mse: mseHistory[mseHistory.length - 1]
                });
            }

            lastRun = {X, y, predictionHistory, predictions, mseHistory, nEstimators};
            drawMainPlots();

            // Calculate global y-axis limits for sequential plot
            const allPredictions = globalPredictionHistory.flatMap(step => step.predictions);
            globalYMin = Math.min(...y, ...allPredictions);
            globalYMax = Math.max(...y, ...allPredictions);
            // Add 10% padding
            const yPadding = (globalYMax - globalYMin) * 0.1;
            globalYMin -= yPadding;
            globalYMax += yPadding;

            // Visualize trees
            visualizeTrees(globalTrees);
            currentStepIndex = 0;
            updateSequentialView();
        }

        function fitSimpleTree(X, residuals, maxDepth) {
            // Simplified tree: creates splits based on quantiles
            const predictions = new Array(X.length).fill(0);
            
            if (maxDepth === 1) {
                const median = calculateMedian(X);
                const leftMean = calculateMean(residuals.filter((_, i) => X[i] <= median));
                const rightMean = calculateMean(residuals.filter((_, i) => X[i] > median));
                
                for (let i = 0; i < X.length; i++) {
                    predictions[i] = X[i] <= median ? leftMean : rightMean;
                }
            } else {
                // Multiple splits for deeper trees
                const splits = Math.pow(2, maxDepth - 1);
                const sortedX = X.slice().sort((a, b) => a - b);
                const thresholds = [];
                
                for (let i = 1; i < splits; i++) {
                    thresholds.push(sortedX[Math.floor(i * X.length / splits)]);
                }
                thresholds.sort((a, b) => a - b);
                
                for (let i = 0; i < X.length; i++) {
                    let region = 0;
                    for (let t of thresholds) {
                        if (X[i] > t) region++;
                    }
                    
                    const inRegion = residuals.filter((_, idx) => {
                        let r = 0;
                        for (let t of thresholds) {
                            if (X[idx] > t) r++;
                        }
                        return r === region;
                    });
                    
                    predictions[i] = calculateMean(inRegion);
                }
            }
            
            return predictions;
        }

        function fitSimpleTreeWithStructure(X, residuals, maxDepth, treeId) {
            const predictions = new Array(X.length).fill(0);
            const sortedX = X.slice().sort((a, b) => a - b);
            
            // Build tree structure
            function buildNode(depth, indices, xMin, xMax) {
                if (depth >= maxDepth || indices.length < 2) {
                    // Leaf node
                    const leafResiduals = indices.map(i => residuals[i]);
                    const value = calculateMean(leafResiduals);
                    return {
                        type: 'leaf',
                        value: value,
                        samples: indices.length
                    };
                }
                
                // Find split point (median of this subset)
                const subsetX = indices.map(i => X[i]).sort((a, b) => a - b);
                const splitValue = subsetX[Math.floor(subsetX.length / 2)];
                
                const leftIndices = indices.filter(i => X[i] <= splitValue);
                const rightIndices = indices.filter(i => X[i] > splitValue);
                
                if (leftIndices.length === 0 || rightIndices.length === 0) {
                    // Can't split, make leaf
                    const leafResiduals = indices.map(i => residuals[i]);
                    const value = calculateMean(leafResiduals);
                    return {
                        type: 'leaf',
                        value: value,
                        samples: indices.length
                    };
                }
                
                return {
                    type: 'split',
                    feature: 'X',
                    threshold: splitValue,
                    left: buildNode(depth + 1, leftIndices, xMin, splitValue),
                    right: buildNode(depth + 1, rightIndices, splitValue, xMax),
                    samples: indices.length
                };
            }
            
            const allIndices = Array.from({length: X.length}, (_, i) => i);
            const tree = {
                id: treeId,
                maxDepth: maxDepth,
                root: buildNode(0, allIndices, Math.min(...X), Math.max(...X))
            };
            
            // Generate predictions from tree
            function predict(node, x) {
                if (node.type === 'leaf') {
                    return node.value;
                }
                if (x <= node.threshold) {
                    return predict(node.left, x);
                } else {
                    return predict(node.right, x);
                }
            }
            
            for (let i = 0; i < X.length; i++) {
                predictions[i] = predict(tree.root, X[i]);
            }
            
            return {
                tree: tree,
                predictions: predictions
            };
        }

        function calculateMSE(y, yPred) {
            return y.reduce((sum, yi, i) => sum + Math.pow(yi - yPred[i], 2), 0) / y.length;
        }

        function calculateMedian(arr) {
            const sorted = arr.slice().sort((a, b) => a - b);
            const mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
        }

        function calculateMean(arr) {
            return arr.length > 0 ? arr.reduce((a, b) => a + b) / arr.length : 0;
        }

        function plotPredictions(X, y, predictionHistory) {
            const traces = [
                {
                    x: X,
                    y: y,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Data',
                    marker: {color: pal().muted, size: 8, opacity: 0.55}
                }
            ];

            // Add initial prediction
            traces.push({
                x: X,
                y: predictionHistory[0],
                mode: 'lines',
                type: 'scatter',
                name: 'Initial constant',
                line: {color: pal().muted, width: 1.5, dash: 'dot'}
            });

            // Add final prediction
            const final = predictionHistory[predictionHistory.length - 1];
            traces.push({
                x: X,
                y: final,
                mode: 'lines',
                type: 'scatter',
                name: 'Ensemble',
                line: {color: pal().accent, width: 2.5}
            });

            const layout = {
                ...baseLayout('Fit against the data'),
                xaxis: axis('x'),
                yaxis: axis('y'),
                hovermode: 'closest',
                showlegend: true,
                legend: {
                    font: {size: 10, family: MONO, color: pal().muted},
                    orientation: 'h',
                    yanchor: 'top',
                    y: -0.15,
                    xanchor: 'center',
                    x: 0.5
                },
                height: 320,
                margin: {l: 50, r: 20, t: 40, b: 70}
            };

            Plotly.newPlot('predictions-plot', traces, layout, {responsive: true, displayModeBar: false});
        }

        function plotResiduals(y, predictions) {
            const residuals = y.map((yi, i) => yi - predictions[i]);
            
            const trace = {
                y: residuals,
                type: 'box',
                name: 'Residuals',
                marker: {color: pal().moss, size: 6}, line: {color: pal().moss, width: 1.5}, fillcolor: pal().mossSoft,
                boxmean: 'sd'
            };

            const layout = {
                ...baseLayout('Residuals'),
                yaxis: axis('residual'),
                showlegend: false,
                height: 320,
                margin: {l: 50, r: 20, t: 40, b: 50}
            };

            Plotly.newPlot('residuals-plot', [trace], layout, {responsive: true, displayModeBar: false});
        }

        function plotLearningCurve(mseHistory) {
            const trace = {
                x: Array.from({length: mseHistory.length}, (_, i) => i),
                y: mseHistory,
                mode: 'lines+markers',
                type: 'scatter',
                name: 'MSE',
                line: {color: pal().accent, width: 2},
                marker: {size: 7, color: pal().accent}
            };

            const layout = {
                ...baseLayout('Training error by round'),
                xaxis: axis('round'),
                yaxis: axis('MSE'),
                showlegend: false,
                height: 320,
                margin: {l: 50, r: 20, t: 40, b: 50}
            };

            Plotly.newPlot('learning-curve', [trace], layout, {responsive: true, displayModeBar: false});
        }

        function displayMetrics(y, predictions, nTrees) {
            const mse = calculateMSE(y, predictions);
            const rmse = Math.sqrt(mse);
            
            const yMean = y.reduce((a, b) => a + b) / y.length;
            const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
            const ssRes = y.reduce((sum, yi, i) => sum + Math.pow(yi - predictions[i], 2), 0);
            const r2 = 1 - (ssRes / ssTot);

            const html = `
                <div class="metric-card">
                    <div class="metric-label">Trees Built</div>
                    <div class="metric-value">${nTrees}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">${rmse.toFixed(3)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-value">${r2.toFixed(3)}</div>
                </div>
            `;

            document.getElementById('metrics-display').innerHTML = html;
        }

        // Visualize multiple trees in a grid
        function visualizeTrees(trees) {
            const container = document.getElementById('trees-grid');
            container.innerHTML = '';
            
            // Define colors for each tree
            const treeColors = pal().leaves;
            
            // Show first 3 trees; one per row when they are wide
            const treesToShow = trees.slice(0, Math.min(3, trees.length));
            const big = treesToShow.some(t => countLeavesOf(t.root) > 8);
            container.classList.toggle('stacked', big);
            
            treesToShow.forEach((tree, idx) => {
                const treeDiv = document.createElement('div');
                treeDiv.className = 'single-tree-view';
                treeDiv.innerHTML = `<h4>Tree ${tree.id + 1}</h4>`;
                
                const svg = document.createElement('div');
                svg.id = `tree-${idx}`;
                treeDiv.appendChild(svg);
                container.appendChild(treeDiv);
                
                drawTree(tree.root, `tree-${idx}`, big ? 900 : 350, 350, treeColors[idx]);
            });
        }

        function countLeavesOf(node) {
            return node.type === 'leaf' ? 1 : countLeavesOf(node.left) + countLeavesOf(node.right);
        }

        // Draw a single tree using D3
        function drawTree(root, containerId, width, height, leafColor = pal().leaves[0]) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            
            // Calculate tree depth to adjust layout
            function getDepth(node) {
                if (node.type === 'leaf') return 1;
                return 1 + Math.max(getDepth(node.left), getDepth(node.right));
            }
            
            function countLeaves(node) {
                return node.type === 'leaf' ? 1 : countLeaves(node.left) + countLeaves(node.right);
            }

            const depth = getDepth(root);
            const leaves = countLeaves(root);
            const adjustedWidth = Math.max(width, leaves * 62);
            height = Math.max(height, 120 + (depth - 1) * 95);
            
            const svg = d3.select(`#${containerId}`)
                .append('svg')
                .attr('width', '100%')
                .attr('height', height)
                .attr('viewBox', `18 0 ${adjustedWidth} ${height}`)
                .attr('preserveAspectRatio', 'xMidYMid meet')
                .style('min-width', adjustedWidth > width ? `${adjustedWidth}px` : null);
            
            const g = svg.append('g')
                .attr('transform', `translate(${adjustedWidth / 2}, 30)`);
            
            // Create tree layout with better spacing
            const treeLayout = d3.tree()
                .size([adjustedWidth - 0.05, height - 125])
                .separation((a, b) => (a.parent == b.parent ? 1 : 1.2));
            
            // Convert to d3 hierarchy
            function toHierarchy(node) {
                if (node.type === 'leaf') {
                    return {
                        name: `${node.value.toFixed(2)}`,
                        type: 'leaf',
                        value: node.value,
                        samples: node.samples
                    };
                }
                return {
                    name: `X ≤ ${node.threshold.toFixed(2)}`,
                    type: 'split',
                    threshold: node.threshold,
                    samples: node.samples,
                    children: [
                        toHierarchy(node.left),
                        toHierarchy(node.right)
                    ]
                };
            }
            
            const hierarchyData = d3.hierarchy(toHierarchy(root));
            const treeData = treeLayout(hierarchyData);
            
            // Center the tree
            const descendants = treeData.descendants();
            const xValues = descendants.map(d => d.x);
            const minX = Math.min(...xValues);
            const maxX = Math.max(...xValues);
            const treeWidth = maxX - minX;
            const offsetX = -treeWidth / 2;
            
            // Draw links (straight lines instead of curved)
            g.selectAll('.tree-link')
                .data(treeData.links())
                .enter()
                .append('line')
                .attr('class', 'tree-link')
                .attr('x1', d => d.source.x + offsetX)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x + offsetX)
                .attr('y2', d => d.target.y)
                .attr('stroke', pal().rule)
                .attr('stroke-width', 2);
            
            // Draw nodes
            const nodes = g.selectAll('.tree-node')
                .data(treeData.descendants())
                .enter()
                .append('g')
                .attr('class', d => `tree-node ${d.data.type}`)
                .attr('transform', d => `translate(${d.x + offsetX},${d.y})`);
            
            nodes.append('title')
                .text(d => d.data.type === 'leaf'
                    ? `leaf: weight ${d.data.value.toFixed(3)}, n = ${d.data.samples}`
                    : `split at x ≤ ${d.data.threshold.toFixed(2)}, n = ${d.data.samples}`);

            nodes.append('circle')
                .attr('r', 22)
                .style('fill', d => d.data.type === 'leaf' ? leafColor : pal().panel)
                .style('stroke', pal().ink)
                .style('stroke-width', d => d.data.type === 'split' ? '2.5px' : '2px');
            
            nodes.append('text')
                .attr('dy', '.35em')
                .attr('text-anchor', 'middle')
                .style('fill', d => d.data.type === 'leaf' ? cssVar('--leaf-ink') : pal().ink)
                .style('font-size', '11px')
                .style('font-weight', '600')
                .text(d => {
                    if (d.data.type === 'leaf') {
                        return d.data.value.toFixed(2);
                    } else {
                        return `≤${d.data.threshold.toFixed(1)}`;
                    }
                });
            
            // Sample counts: below leaves, beside internal nodes
            nodes.append('text')
                .attr('dx', d => d.data.type === 'leaf' ? 0 : 26)
                .attr('dy', d => d.data.type === 'leaf' ? '4.9em' : '-0.9em')
                .attr('text-anchor', d => d.data.type === 'leaf' ? 'middle' : 'start')
                .style('font-size', '10px')
                .style('fill', pal().muted)
                .text(d => `n=${d.data.samples}`);

            // Deep trees overflow sideways; start centered on the root
            if (adjustedWidth > width) {
                const scroller = container.closest('.tree-container, .single-tree-view') || container;
                requestAnimationFrame(() => {
                    scroller.scrollLeft = (scroller.scrollWidth - scroller.clientWidth) / 2;
                });
            }
        }

        // Sequential step through
        function updateSequentialView() {
            const step = globalPredictionHistory[currentStepIndex];
            
            // Update button states
            const prevBtn = document.getElementById('prev-btn');
            const nextBtn = document.getElementById('next-btn');
            
            prevBtn.disabled = currentStepIndex === 0;
            nextBtn.disabled = currentStepIndex === globalPredictionHistory.length - 1;
            
            // Update step display
            if (currentStepIndex === 0) {
                document.getElementById('step-display').textContent = 'Round 0: initial constant';
            } else {
                document.getElementById('step-display').textContent = `Round ${currentStepIndex}: after tree ${currentStepIndex}`;
            }
            
            // Draw current tree
            const container = document.getElementById('sequential-tree-view');
            container.innerHTML = '';
            
            const seqGrid = container.parentElement;
            seqGrid.classList.toggle('stacked', !!step.tree && countLeavesOf(step.tree.root) > 8);

            if (step.tree) {
                container.innerHTML = `<h4>Tree ${step.tree.id + 1}</h4>`;
                const treeDiv = document.createElement('div');
                treeDiv.id = 'current-tree-viz';
                treeDiv.style.width = '100%';
                treeDiv.style.minHeight = '380px';
                container.appendChild(treeDiv);
                
                // Cycle through colors
                const treeColors = pal().leaves;
                const colorIndex = step.tree.id % 3;
                drawTree(step.tree.root, 'current-tree-viz', seqGrid.classList.contains('stacked') ? 900 : 500, 380, treeColors[colorIndex]);
            } else {
                container.innerHTML = `
                    <h4>Before the first tree</h4>
                    <div class="empty">
                        <p>Nothing has been grown yet. The prediction is the mean of the targets, the same for every sample.</p>
                    </div>
                `;
            }
            
            // Update predictions plot
            plotSequentialPredictions(currentStepIndex);
        }

        function previousStep() {
            if (currentStepIndex > 0) {
                currentStepIndex--;
                updateSequentialView();
            }
        }

        function nextStep() {
            if (currentStepIndex < globalPredictionHistory.length - 1) {
                currentStepIndex++;
                updateSequentialView();
            }
        }

        function plotSequentialPredictions(stepIndex) {
            if (globalX.length === 0) return;
            
            const step = globalPredictionHistory[stepIndex];
            
            const traces = [
                {
                    x: globalX,
                    y: globalY,
                    mode: 'markers',
                    type: 'scatter',
                    name: 'Data',
                    marker: {color: pal().muted, size: 7, opacity: 0.55}
                },
                {
                    x: globalX,
                    y: step.predictions,
                    mode: 'lines',
                    type: 'scatter',
                    name: 'Ensemble so far',
                    line: {color: pal().accent, width: 2.5}
                }
            ];

            const layout = {
                ...baseLayout(`Round ${stepIndex} · MSE ${step.mse.toFixed(3)}`),
                xaxis: axis('x'),
                yaxis: axis('y'),
                hovermode: 'closest',
                showlegend: true,
                legend: {
                    font: {size: 10, family: MONO, color: pal().muted},
                    orientation: 'h',
                    yanchor: 'top',
                    y: -0.2,
                    xanchor: 'center',
                    x: 0.5
                },
                height: 400,
                margin: {l: 60, r: 30, t: 50, b: 90}
            };

            Plotly.newPlot('sequential-predictions-plot', traces, layout, {responsive: true, displayModeBar: false});
        }

        // Run demo on page load
        window.addEventListener('load', runXGBoostDemo);


        // Keyboard: arrows step rounds, d toggles theme
        document.addEventListener('keydown', e => {
            const tag = (e.target.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' || e.metaKey || e.ctrlKey || e.altKey) return;
            if (e.key === 'ArrowRight') { nextStep(); }
            else if (e.key === 'ArrowLeft') { previousStep(); }
            else if (e.key === 'd' || e.key === 'D') { document.getElementById('themeToggle').click(); }
        });

        // Reading progress
        (function () {
            const bar = document.getElementById('progress');
            function update() {
                const max = document.documentElement.scrollHeight - window.innerHeight;
                bar.style.width = (max > 0 ? (window.scrollY / max) * 100 : 0) + '%';
            }
            window.addEventListener('scroll', update, {passive: true});
            window.addEventListener('resize', update);
            update();
        })();

        // Light / dark toggle
        const themeToggle = document.getElementById('themeToggle');
        themeToggle.addEventListener('click', () => {
            const root = document.documentElement;
            const dark = root.getAttribute('data-theme') === 'dark';
            if (dark) root.removeAttribute('data-theme'); else root.setAttribute('data-theme', 'dark');
            localStorage.setItem('theme', dark ? 'light' : 'dark');
            redrawForTheme();
        });

        // Contents panel
        const hamburger = document.getElementById('hamburger');
        const navMenu = document.getElementById('navMenu');
        const overlay = document.getElementById('overlay');
        const navLinks = Array.from(document.querySelectorAll('.nav-menu a'));

        function toggleMenu(force) {
            const open = typeof force === 'boolean' ? force : !navMenu.classList.contains('active');
            hamburger.classList.toggle('active', open);
            navMenu.classList.toggle('active', open);
            overlay.classList.toggle('active', open);
            hamburger.setAttribute('aria-expanded', String(open));
        }

        hamburger.addEventListener('click', () => toggleMenu());
        overlay.addEventListener('click', () => toggleMenu(false));
        document.addEventListener('keydown', e => { if (e.key === 'Escape') toggleMenu(false); });

        navLinks.forEach(link => {
            link.addEventListener('click', e => {
                e.preventDefault();
                const target = document.getElementById(link.getAttribute('href').slice(1));
                if (target) {
                    target.scrollIntoView({behavior: 'smooth', block: 'start'});
                    toggleMenu(false);
                }
            });
        });

        // Mark the current section in the contents panel
        (function () {
            const sections = navLinks
                .map(a => document.getElementById(a.getAttribute('href').slice(1)))
                .filter(Boolean);
            if (!('IntersectionObserver' in window) || !sections.length) return;
            const byId = new Map(navLinks.map(a => [a.getAttribute('href').slice(1), a]));
            const observer = new IntersectionObserver(entries => {
                entries.forEach(entry => {
                    if (!entry.isIntersecting) return;
                    navLinks.forEach(a => a.classList.remove('is-active'));
                    const link = byId.get(entry.target.id);
                    if (link) link.classList.add('is-active');
                });
            }, {rootMargin: '-10% 0px -70% 0px'});
            sections.forEach(sec => observer.observe(sec));
        })();
