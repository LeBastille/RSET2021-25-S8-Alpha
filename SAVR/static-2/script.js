document.addEventListener('DOMContentLoaded', function () {
    // Feature Cards
    const features = [
        {
            icon: '📊',
            title: 'Price History',
            description: 'Track price changes over time and get notified when prices drop.'
        },
        {
            icon: '⚖️',
            title: 'Side-by-Side Comparison',
            description: 'Compare features, prices, and reviews across multiple products.'
        },
        {
            icon: '🏷️',
            title: 'Deal Alerts',
            description: 'Set up alerts for price drops on your favorite products.'
        }
    ];

    function createFeatureCard(feature) {
        return `
            <div class="feature-card">
                <div class="feature-icon">${feature.icon}</div>
                <h3>${feature.title}</h3>
                <p>${feature.description}</p>
            </div>
        `;
    }

    const featuresContainer = document.getElementById('features-container');
    if (featuresContainer) {
        features.forEach(feature => {
            featuresContainer.innerHTML += createFeatureCard(feature);
        });
    }

    // Dummy Products for Testing
    const dummyProducts = ["Product A", "Product B"];

    // Search functionality with dropdown suggestions
    const searchBar = document.querySelector('.search-bar');
    const compareBtn = document.querySelector('.compare-btn');
    const dropdown = document.getElementById("search-dropdown");

    searchBar.addEventListener("input", function () {
        const searchTerm = searchBar.value.trim().toLowerCase();

        dropdown.innerHTML = "";
        if (searchTerm.length < 1) {
            dropdown.style.display = "none";
            return;
        }

        const filteredProducts = dummyProducts.filter(product => 
            product.toLowerCase().includes(searchTerm)
        );

        if (filteredProducts.length === 0) {
            dropdown.style.display = "none";
            return;
        }

        filteredProducts.forEach(item => {
            const option = document.createElement("div");
            option.classList.add("dropdown-item");
            option.textContent = item;
            option.addEventListener("click", () => {
                searchBar.value = item;
                dropdown.style.display = "none";
            });
            dropdown.appendChild(option);
        });

        dropdown.style.display = "block";
    });

    document.addEventListener("click", function (e) {
        if (!searchBar.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.style.display = "none";
        }
    });

    // Product Comparison Functionality
    function compareProducts() {
        const product1 = document.getElementById("product1")?.value;
        const product2 = document.getElementById("product2")?.value;

        if (!product1 || !product2 || product1 === product2) {
            alert("Please select two different products.");
            return;
        }

        fetch("/compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ products: [product1, product2] })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            const results = data.comparison.map(item => `
                <p><strong>Rank ${item.rank}:</strong> ${item.name} (Score: ${item.score})</p>
            `).join("");

            document.getElementById("comparison-results").innerHTML = `
                <h2>Comparison Results</h2>
                ${results}
            `;
        })
        .catch(error => console.error("Error:", error));
    }

    // Attach Compare Button event listener
    if (compareBtn) {
        compareBtn.addEventListener('click', compareProducts);
    }
});
