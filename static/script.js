// Toggle hamburger dropdown menu
function toggleMenu() {
    const dropdown = document.getElementById("dropdownMenu");
    dropdown.classList.toggle("show");
}

// Modal image viewer logic
document.addEventListener("DOMContentLoaded", function () {
    const modal = document.getElementById("imageModal");
    const modalImg = document.getElementById("modalImg");
    const captionText = document.getElementById("caption");
    const closeBtn = document.getElementsByClassName("close")[0];
    const forecastImages = document.querySelectorAll(".forecast-image, .forecast-item img");

    forecastImages.forEach(img => {
        img.style.cursor = "pointer"; // Add pointer on hover
        img.addEventListener("click", function () {
            modal.style.display = "block";
            modalImg.src = this.src;
            captionText.textContent = this.alt;
        });
    });

    closeBtn.onclick = function () {
        modal.style.display = "none";
    };

    modal.onclick = function (e) {
        if (e.target === modal) {
            modal.style.display = "none";
        }
    };
    // // === Fetch and process CSV data ===
    // console.log('Fetching CSV data...');
    // fetch('/static/merged_file_2.csv')  // Ensure the path to your CSV file is correct
    //     .then(response => {
    //         if (!response.ok) {
    //             throw new Error('Failed to fetch the CSV file.');
    //         }
    //         return response.text();
    //     })
    //     .then(csv => {
    //         console.log('CSV file fetched successfully!');
            
    //         const parsed = Papa.parse(csv, { header: true });
    //         console.log('Parsed CSV:', parsed);  // Log parsed data
            
    //         const data = parsed.data;
    //         const monthlySales = {};

    //         data.forEach(row => {
    //             // Extract the month from the InvoiceDate (e.g., "2025-01-31")
    //             const date = new Date(row.InvoiceDate);  // Assumes date is in the format "YYYY-MM-DD"
    //             const month = date.toLocaleString('default', { month: 'short' });  // Converts to "Jan", "Feb", etc.
    //             const year = date.getFullYear();  // Get the year, in case you want to use it
    //             const key = `${month}-${year}`;  // Create a key like "Jan-2025"

    //             // Aggregate sales (Taxable Amount) for the corresponding month
    //             if (!monthlySales[key]) monthlySales[key] = 0;
    //             monthlySales[key] += parseFloat(row['Taxable Amount'] || 0);  // Use 'Taxable Amount' as sales
    //         });

    //         console.log('Aggregated monthly sales:', monthlySales);  // Log the aggregated sales data

    //         // Update the table with the aggregated monthly sales data
    //         for (const key in monthlySales) {
    //             const [month] = key.split('-');  // Get the month part (e.g., "Jan")
    //             const cell = document.querySelector(`td[data-month="${month}"]`);  // Find the corresponding table cell

    //             if (cell) {
    //                 console.log(`Updating cell for ${month} with value: ${monthlySales[key]}`);
    //                 cell.textContent = monthlySales[key].toFixed(2);  // Update the cell with the sales value
    //             }
    //         }
    //     })
    //     .catch(error => {
    //         console.error('Error:', error);  // Catch any errors
    //     });
});
