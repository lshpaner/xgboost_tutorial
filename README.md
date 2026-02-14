# XGBoost Interactive Guide

An interactive web-based educational tool explaining how XGBoost (Extreme Gradient Boosting) works.

## Features

- **Interactive Demo**: Adjust parameters (number of trees, learning rate, max depth) in real-time
- **Visual Tree Structures**: See actual decision trees built by XGBoost with pastel color coding
- **Sequential Step-Through**: Watch how predictions improve tree-by-tree
- **Mathematical Explanations**: LaTeX-rendered formulas explaining the theory
- **Comparison Table**: XGBoost vs Random Forest vs Gradient Boosting
- **Mobile Responsive**: Works on desktop and mobile devices

## Project Structure

```
xgboost_demo/
├── index.html      # Main HTML structure
├── styles.css      # All styling and responsive design
├── script.js       # XGBoost simulation and visualization logic
└── README.md       # This file
```

## Technologies Used

- **Plotly.js**: Interactive plotting
- **D3.js v7**: Tree visualization
- **MathJax**: LaTeX math rendering
- **Math.js**: Mathematical operations
- **Vanilla JavaScript**: No frameworks

## Deployment

### Option 1: GitHub Pages
1. Create a new GitHub repository
2. Upload all files
3. Go to Settings > Pages
4. Select main branch
5. Your site will be live at `https://username.github.io/repo-name`

### Option 2: Netlify
1. Drag and drop the entire folder to [Netlify](https://app.netlify.com/drop)
2. Instant deployment with custom URL

### Option 3: Local
Simply open `index.html` in any modern web browser

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari
- Mobile browsers (iOS Safari, Chrome Mobile)

## Credits

Created by Leon Shpaner  
© 2025

## License

Feel free to use for educational purposes.
