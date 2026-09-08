"""Market universe dataset containing NASDAQ-100 constituents and S&P 500 sector titans.

Provides curated node metadata and relational edges (competitors, suppliers, customers,
datacenter power/infrastructure partners, and index benchmarks) for the Stock Relationship Graph.
"""
from __future__ import annotations

from typing import List
from core.stock_graph import StockGraph, StockNode, GraphEdge, RelationType


def get_nasdaq_100_tickers() -> List[str]:
    """Return the list of NASDAQ-100 tickers."""
    return [
        "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
        "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AXON", "BIIB", "BKNG", "BKR",
        "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
        "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DLTR", "DXCM", "EA", "EXC", "EXPE",
        "FAST", "FSLR", "FTNT", "GEHC", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN",
        "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR",
        "MCHP", "MDB", "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL", "MSFT", "MU",
        "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD",
        "PEP", "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS",
        "TEAM", "TMUS", "TSLA", "TTD", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL",
        "ZS"
    ]


def get_sp500_titans() -> List[str]:
    """Return the list of major non-NASDAQ S&P 500 sector leaders."""
    return [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "BRK.B", "V", "MA", "AXP", "BLK", "SCHW", "CB", "PGR",
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY", "CVS", "ELV", "MDT", "NVO",
        "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "KMI", "WMB", "EQT", "CCJ",
        "CAT", "GE", "LMT", "RTX", "BA", "DE", "UNP", "UPS", "FDX", "ETN", "EMR", "NOC", "GD", "VRT",
        "WMT", "HD", "MCD", "NKE", "LOW", "TGT", "TJX", "GM", "F", "UBER", "DIS",
        "PG", "KO", "PM", "MO", "CL",
        "CRM", "ORCL", "IBM", "NOW", "DELL", "HPE", "HPQ", "ACN", "SNOW",
        "NEE", "SO", "DUK", "VST", "SRE", "D",
        "EQIX", "DLR", "PLD", "AMT", "CCI", "SPG",
        "SHW", "FCX", "NEM", "NUE", "TSM", "COIN", "MSTR",
        "SPY", "QQQ", "DIA", "IWM", "SMH", "XLF", "XLE", "XLV", "XLI", "XLU", "XLK", "XLP", "XLY"
    ]


def get_market_nodes() -> List[StockNode]:
    """Return all curated StockNodes across NASDAQ-100 and S&P 500 titans."""
    return [
        # --- TECHNOLOGY & SEMICONDUCTORS (NASDAQ-100) ---
        StockNode("NVDA", "NVIDIA Corp", "Semiconductors", "Compute & Networking", "Mega", "AI accelerator GPU market leader & CUDA ecosystem"),
        StockNode("AMD", "Advanced Micro Devices", "Semiconductors", "Compute & Graphics", "Large", "Direct x86 CPU and datacenter GPU competitor"),
        StockNode("INTC", "Intel Corp", "Semiconductors", "Processors & Foundries", "Large", "x86 CPU vendor and emerging US commercial foundry"),
        StockNode("AVGO", "Broadcom Inc", "Semiconductors", "Custom Silicon & Networking", "Mega", "Custom AI ASICs (TPU/Trainium) & datacenter switches"),
        StockNode("QCOM", "Qualcomm Inc", "Semiconductors", "Wireless & Edge AI", "Large", "Snapdragon mobile processors and edge AI NPU silicon"),
        StockNode("TXN", "Texas Instruments", "Semiconductors", "Analog & Embedded", "Large", "Industrial and automotive analog integrated circuits"),
        StockNode("ADI", "Analog Devices", "Semiconductors", "Analog & Mixed Signal", "Large", "High-performance analog, power management, and sensor interfaces"),
        StockNode("AMAT", "Applied Materials", "Semiconductors", "Wafer Fab Equipment", "Large", "Global leader in materials engineering & wafer fabrication gear"),
        StockNode("LRCX", "Lam Research", "Semiconductors", "Etch & Deposition", "Large", "Specialist in wafer etch, deposition, and 3D NAND equipment"),
        StockNode("KLAC", "KLA Corp", "Semiconductors", "Process Control & Yield", "Large", "Yield management and inspection systems for advanced nodes"),
        StockNode("ASML", "ASML Holding", "Semiconductors", "Lithography Systems", "Mega", "Sole global supplier of EUV and High-NA lithography systems"),
        StockNode("MU", "Micron Technology", "Semiconductors", "Memory & Storage", "Large", "HBM3e high-bandwidth memory and DRAM for AI accelerators"),
        StockNode("NXPI", "NXP Semiconductors", "Semiconductors", "Automotive & Industrial", "Large", "Automotive processing, secure connectivity, and radar chips"),
        StockNode("MRVL", "Marvell Technology", "Semiconductors", "Data Infrastructure", "Large", "Optical interconnects, custom ASICs, and storage controllers"),
        StockNode("MCHP", "Microchip Technology", "Semiconductors", "Microcontrollers & Mixed Signal", "Large", "Embedded microcontrollers, FPGA, and analog components"),
        StockNode("ON", "ON Semiconductor", "Semiconductors", "Power & Silicon Carbide", "Large", "Silicon carbide (SiC) power modules for EVs and energy"),
        StockNode("ARM", "Arm Holdings plc", "Semiconductors", "Processor IP Licensing", "Large", "Architecture IP licensor for mobile, server, and IoT chips"),
        StockNode("TSM", "Taiwan Semiconductor", "Semiconductors", "Pure-Play Foundry", "Mega", "World's primary contract manufacturer for advanced chips"),

        # --- SOFTWARE, SAAS & ENTERPRISE TECH (NASDAQ-100) ---
        StockNode("MSFT", "Microsoft Corp", "Technology", "Cloud & Enterprise Software", "Mega", "Azure cloud hyperscaler, Windows, Office 365 & OpenAI backer"),
        StockNode("AAPL", "Apple Inc", "Technology", "Consumer Electronics", "Mega", "iPhone, Mac, iPad consumer ecosystem & Apple Intelligence"),
        StockNode("ADBE", "Adobe Inc", "Technology", "Creative & Document Cloud", "Large", "Creative Cloud suite, Firefly AI, and digital marketing tools"),
        StockNode("INTU", "Intuit Inc", "Technology", "Financial Software", "Large", "TurboTax, QuickBooks, Credit Karma, and Mailchimp"),
        StockNode("PANW", "Palo Alto Networks", "Technology", "Cybersecurity", "Large", "Next-gen enterprise firewall, Prisma Cloud, and SASE security"),
        StockNode("CRWD", "CrowdStrike Holdings", "Technology", "Endpoint Security", "Large", "Falcon cloud-native endpoint protection & threat intelligence"),
        StockNode("FTNT", "Fortinet Inc", "Technology", "Network Security", "Large", "FortiGate hardware firewalls and integrated cybersecurity"),
        StockNode("ZS", "Zscaler Inc", "Technology", "Zero Trust Cloud Security", "Large", "Zero Trust Exchange cloud platform for enterprise access"),
        StockNode("TEAM", "Atlassian Corp", "Technology", "Collaboration Software", "Large", "Jira, Confluence, and developer workflow collaboration"),
        StockNode("SNPS", "Synopsys Inc", "Technology", "Electronic Design Automation", "Large", "Silicon design software (EDA), semiconductor IP & software integrity"),
        StockNode("CDNS", "Cadence Design Systems", "Technology", "Electronic Design Automation", "Large", "Computational software for integrated circuit and PCB design"),
        StockNode("ANSS", "ANSYS Inc", "Technology", "Engineering Simulation", "Large", "Engineering multiphysics simulation and modeling software"),
        StockNode("ADSK", "Autodesk Inc", "Technology", "Design & Engineering Software", "Large", "AutoCAD, Revit, Maya 3D design and engineering software"),
        StockNode("MDB", "MongoDB Inc", "Technology", "Database Platform", "Large", "Modern document-based developer data platform and Atlas cloud"),
        StockNode("DDOG", "Datadog Inc", "Technology", "Cloud Observability", "Large", "Monitoring and security platform for cloud applications and infra"),
        StockNode("WDAY", "Workday Inc", "Technology", "Human Capital Management", "Large", "Cloud enterprise applications for finance and human resources"),
        StockNode("PLTR", "Palantir Technologies", "Technology", "AI Platforms & Analytics", "Large", "Foundry, Gotham, and AIP enterprise decision-making software"),
        StockNode("APP", "AppLovin Corp", "Technology", "Mobile AdTech Software", "Large", "AI-driven mobile advertising and app monetization software"),
        StockNode("CDW", "CDW Corp", "Technology", "IT Solutions & Hardware", "Large", "Technology products and services for business, gov, and education"),
        StockNode("CTSH", "Cognizant Technology", "Technology", "IT Consulting & Services", "Large", "Digital transformation and IT systems integration services"),
        StockNode("ROP", "Roper Technologies", "Technology", "Vertical Software", "Large", "Diversified software and engineered product solutions"),
        StockNode("CSCO", "Cisco Systems", "Technology", "Networking Hardware", "Large", "Enterprise networking switches, routers, and Splunk observability"),
        StockNode("SMCI", "Super Micro Computer", "Technology Hardware", "AI Server Systems", "Large", "High-density GPU server racks and direct liquid cooling solutions"),

        # --- S&P 500 TECH TITANS ---
        StockNode("CRM", "Salesforce Inc", "Technology", "Enterprise CRM & Agentic AI", "Large", "Sales Cloud, Service Cloud, Slack, and Agentforce AI platform"),
        StockNode("ORCL", "Oracle Corp", "Technology", "Database & OCI Cloud", "Mega", "Autonomous Database and Oracle Cloud Infrastructure (OCI) AI clusters"),
        StockNode("IBM", "IBM Corp", "Technology", "Hybrid Cloud & Enterprise AI", "Large", "Red Hat OpenShift, mainframe systems, and watsonx enterprise AI"),
        StockNode("NOW", "ServiceNow Inc", "Technology", "Workflow Automation", "Large", "Now Platform for digital enterprise workflow orchestration"),
        StockNode("DELL", "Dell Technologies", "Technology Hardware", "Enterprise AI Hardware", "Large", "PowerEdge AI servers, storage arrays, and enterprise PCs"),
        StockNode("HPE", "Hewlett Packard Enterprise", "Technology Hardware", "Hybrid Cloud & Supercomputing", "Large", "Cray supercomputers, ProLiant servers, and GreenLake cloud"),
        StockNode("HPQ", "HP Inc", "Technology Hardware", "Personal Systems & Printing", "Large", "Consumer and commercial PCs, workstations, and printers"),
        StockNode("ACN", "Accenture plc", "Technology", "IT Consulting & Services", "Large", "Global management consulting and technology implementation"),
        StockNode("SNOW", "Snowflake Inc", "Technology", "Cloud Data Platform", "Large", "Data cloud, data warehousing, and AI analytics platform"),

        # --- COMMUNICATION SERVICES & MEDIA ---
        StockNode("GOOGL", "Alphabet Inc (Class A)", "Communication Services", "Search & Cloud", "Mega", "Google Search, YouTube, Android, Google Cloud, and Gemini AI"),
        StockNode("GOOG", "Alphabet Inc (Class C)", "Communication Services", "Search & Cloud", "Mega", "Google non-voting equity share class"),
        StockNode("META", "Meta Platforms", "Communication Services", "Social Media & Open AI", "Mega", "Instagram, Facebook, WhatsApp, Reality Labs & Llama open models"),
        StockNode("NFLX", "Netflix Inc", "Communication Services", "Streaming Entertainment", "Mega", "Global subscription video on demand and original content"),
        StockNode("DIS", "The Walt Disney Co", "Communication Services", "Entertainment & Media", "Mega", "Disney+, theme parks, ESPN, and film entertainment studios"),
        StockNode("TMUS", "T-Mobile US", "Communication Services", "Wireless Telecommunications", "Large", "5G mobile network carrier and broadband provider"),
        StockNode("CMCSA", "Comcast Corp", "Communication Services", "Cable & Media", "Large", "Xfinity broadband, NBCUniversal media, and theme parks"),
        StockNode("CHTR", "Charter Communications", "Communication Services", "Broadband Connectivity", "Large", "Spectrum broadband, cable television, and mobile services"),
        StockNode("WBD", "Warner Bros. Discovery", "Communication Services", "Media & Entertainment", "Large", "Warner Bros studios, HBO, Max streaming, and discovery channels"),
        StockNode("EA", "Electronic Arts", "Communication Services", "Video Game Publishing", "Large", "EA SPORTS FC, Madden, Apex Legends, and interactive gaming"),
        StockNode("TTD", "The Trade Desk", "Communication Services", "Digital AdTech", "Large", "Independent programmatic demand-side ad-buying platform"),

        # --- CONSUMER DISCRETIONARY & RETAIL ---
        StockNode("AMZN", "Amazon.com Inc", "Consumer Discretionary", "E-Commerce & AWS", "Mega", "Global e-commerce marketplace, logistics network, and AWS cloud"),
        StockNode("TSLA", "Tesla Inc", "Consumer Discretionary", "Electric Vehicles & Energy", "Mega", "EV manufacturing, energy storage, Full Self-Driving & Optimus"),
        StockNode("BKNG", "Booking Holdings", "Consumer Discretionary", "Online Travel Agencies", "Large", "Booking.com, Priceline, Agoda, and Kayak travel booking"),
        StockNode("SBUX", "Starbucks Corp", "Consumer Discretionary", "Specialty Coffee Retail", "Large", "Global chain of specialty coffeehouses and roast reserves"),
        StockNode("MAR", "Marriott International", "Consumer Discretionary", "Hotels & Lodging", "Large", "Marriott, Sheraton, Ritz-Carlton hotel brands & Bonvoy"),
        StockNode("ORLY", "O'Reilly Automotive", "Consumer Discretionary", "Automotive Parts", "Large", "Aftermarket parts, tools, and maintenance supplies for vehicles"),
        StockNode("CPRT", "Copart Inc", "Consumer Discretionary", "Vehicle Auctions", "Large", "Online vehicle remarketing and salvage auction services"),
        StockNode("LULU", "Lululemon Athletica", "Consumer Discretionary", "Athletic Apparel", "Large", "Technical athletic apparel and footwear for fitness and lifestyle"),
        StockNode("ROST", "Ross Stores", "Consumer Discretionary", "Off-Price Retail", "Large", "Off-price apparel and home fashion retail chain"),
        StockNode("DASH", "DoorDash Inc", "Consumer Discretionary", "On-Demand Delivery", "Large", "Food and convenience delivery platform and logistics network"),
        StockNode("ABNB", "Airbnb Inc", "Consumer Discretionary", "Short-Term Rentals", "Large", "Global marketplace for stays, short-term rentals, and experiences"),
        StockNode("EXPE", "Expedia Group", "Consumer Discretionary", "Online Travel", "Large", "Expedia, Hotels.com, Vrbo, and Orbitz travel platforms"),
        StockNode("MELI", "MercadoLibre Inc", "Consumer Discretionary", "E-Commerce & Fintech", "Large", "Leading Latin American e-commerce ecosystem and Mercado Pago"),
        StockNode("PDD", "PDD Holdings", "Consumer Discretionary", "Discount Marketplace", "Large", "Pinduoduo social e-commerce and Temu cross-border marketplace"),
        StockNode("HD", "The Home Depot", "Consumer Discretionary", "Home Improvement Retail", "Large", "Home improvement retailer supplying DIY and professional contractors"),
        StockNode("LOW", "Lowe's Companies", "Consumer Discretionary", "Home Improvement Retail", "Large", "Home improvement, renovation, and maintenance products"),
        StockNode("MCD", "McDonald's Corp", "Consumer Discretionary", "Fast Food Restaurants", "Large", "World's largest fast-food restaurant chain franchise system"),
        StockNode("NKE", "Nike Inc", "Consumer Discretionary", "Athletic Footwear & Apparel", "Large", "Athletic footwear, apparel, sports equipment, and Jordan brand"),
        StockNode("TGT", "Target Corp", "Consumer Discretionary", "General Merchandise Retail", "Large", "Discount department store chain offering apparel, home, and food"),
        StockNode("TJX", "The TJX Companies", "Consumer Discretionary", "Off-Price Retail", "Large", "TJ Maxx, Marshalls, HomeGoods off-price apparel and home goods"),
        StockNode("GM", "General Motors", "Consumer Discretionary", "Automobile Manufacturer", "Large", "Automotive manufacturing, Chevrolet, GMC, Cadillac, and Cruise"),
        StockNode("F", "Ford Motor Co", "Consumer Discretionary", "Automobile Manufacturer", "Large", "F-Series pickup trucks, commercial vehicles, and Ford Pro fleet"),
        StockNode("UBER", "Uber Technologies", "Consumer Discretionary", "Mobility & Delivery", "Large", "Ridesharing, Uber Eats delivery, and freight logistics network"),

        # --- CONSUMER STAPLES ---
        StockNode("COST", "Costco Wholesale", "Consumer Staples", "Membership Warehouses", "Mega", "Membership wholesale club operator selling bulk merchandise"),
        StockNode("PEP", "PepsiCo Inc", "Consumer Staples", "Snacks & Beverages", "Mega", "Lay's, Doritos, Pepsi, Gatorade, and Quaker foods"),
        StockNode("MDLZ", "Mondelez International", "Consumer Staples", "Confectionery & Snacks", "Large", "Oreo, Cadbury, Ritz, and Toblerone snack brands"),
        StockNode("KDP", "Keurig Dr Pepper", "Consumer Staples", "Beverages & Coffee", "Large", "Dr Pepper, 7UP, Snapple, and Keurig single-serve coffee"),
        StockNode("MNST", "Monster Beverage", "Consumer Staples", "Energy Drinks", "Large", "Energy drinks and alternative beverages distributed via Coca-Cola"),
        StockNode("KHC", "The Kraft Heinz Co", "Consumer Staples", "Packaged Food", "Large", "Condiments, sauces, meals, cheese, and packaged grocery goods"),
        StockNode("CCEP", "Coca-Cola Europacific Partners", "Consumer Staples", "Beverage Bottling", "Large", "Major independent bottler of Coca-Cola products in Europe/APAC"),
        StockNode("DLTR", "Dollar Tree", "Consumer Staples", "Discount Retail", "Large", "Discount variety stores and Family Dollar retail chains"),
        StockNode("WMT", "Walmart Inc", "Consumer Staples", "Hypermarket Retail", "Mega", "World's largest retailer operating supercenters, discount stores & e-commerce"),
        StockNode("PG", "Procter & Gamble", "Consumer Staples", "Household & Personal Care", "Mega", "Tide, Pampers, Gillette, Crest, and consumer packaged goods"),
        StockNode("KO", "The Coca-Cola Co", "Consumer Staples", "Non-Alcoholic Beverages", "Mega", "Coca-Cola, Sprite, Fanta, Dasani, and global beverage licensing"),
        StockNode("PM", "Philip Morris International", "Consumer Staples", "Tobacco & Smoke-Free", "Large", "Marlboro international, IQOS heated tobacco, and ZYN nicotine pouches"),
        StockNode("MO", "Altria Group", "Consumer Staples", "Tobacco & Smoke-Free", "Large", "Marlboro domestic US cigarettes, Copenhagen, and on! pouches"),
        StockNode("CL", "Colgate-Palmolive", "Consumer Staples", "Oral Care & Household", "Large", "Colgate oral hygiene, Palmolive, Protex, and Hill's Pet Nutrition"),

        # --- HEALTHCARE & BIOTECH ---
        StockNode("AMGN", "Amgen Inc", "Healthcare", "Biotechnology", "Large", "Biopharmaceuticals for oncology, cardiovascular disease, and bone health"),
        StockNode("ISRG", "Intuitive Surgical", "Healthcare", "Robotic Surgical Systems", "Large", "da Vinci robotic-assisted minimally invasive surgical systems"),
        StockNode("VRTX", "Vertex Pharmaceuticals", "Healthcare", "Biopharmaceuticals", "Large", "Cystic fibrosis therapies and CRISPR gene editing (Casgevy)"),
        StockNode("GILD", "Gilead Sciences", "Healthcare", "Antiviral Biopharma", "Large", "HIV antiretroviral therapies, hepatitis treatments, and oncology cell therapy"),
        StockNode("REGN", "Regeneron Pharmaceuticals", "Healthcare", "Monoclonal Antibodies", "Large", "Dupixent, Eylea, and antibody discovery platforms"),
        StockNode("DXCM", "DexCom Inc", "Healthcare", "Continuous Glucose Monitors", "Large", "Continuous glucose monitoring (CGM) systems for diabetes management"),
        StockNode("IDXX", "IDEXX Laboratories", "Healthcare", "Veterinary Diagnostics", "Large", "Diagnostic test kits, instruments, and software for animal health"),
        StockNode("BIIB", "Biogen Inc", "Healthcare", "Neuroscience Biopharma", "Large", "Neurological therapies for Alzheimer's (Leqembi) and multiple sclerosis"),
        StockNode("GEHC", "GE HealthCare", "Healthcare", "Medical Imaging & Monitoring", "Large", "MRI, CT scanners, ultrasound equipment, and patient monitoring"),
        StockNode("ILMN", "Illumina Inc", "Healthcare", "Genomics & Sequencing", "Large", "DNA sequencing and array-based technologies for genetic research"),
        StockNode("MRNA", "Moderna Inc", "Healthcare", "mRNA Therapeutics", "Large", "mRNA vaccines and individualized neoantigen cancer therapies"),
        StockNode("LLY", "Eli Lilly and Co", "Healthcare", "Pharmaceuticals & GLP-1", "Mega", "Mounjaro, Zepbound GLP-1/GIP receptor agonists and oncology"),
        StockNode("NVO", "Novo Nordisk", "Healthcare", "Diabetes & Obesity Care", "Mega", "Ozempic and Wegovy semaglutide GLP-1 therapies"),
        StockNode("UNH", "UnitedHealth Group", "Healthcare", "Managed Healthcare", "Mega", "UnitedHealthcare insurance plans and Optum healthcare services"),
        StockNode("JNJ", "Johnson & Johnson", "Healthcare", "Innovative Medicine & Devices", "Mega", "Pharmaceutical therapies (Darzalex, Stelara) and medical devices"),
        StockNode("ABBV", "AbbVie Inc", "Healthcare", "Immunology & Aesthetics", "Large", "Humira, Skyrizi, Rinvoq immunology drugs and Botox aesthetics"),
        StockNode("MRK", "Merck & Co", "Healthcare", "Immuno-Oncology & Vaccines", "Large", "Keytruda anti-PD-1 cancer therapy, Gardasil, and animal health"),
        StockNode("PFE", "Pfizer Inc", "Healthcare", "Biopharmaceuticals", "Large", "Vaccines, oncology treatments (Seagen acquisition), and cardiovascular"),
        StockNode("TMO", "Thermo Fisher Scientific", "Healthcare", "Life Science Instruments", "Large", "Analytical instruments, laboratory reagents, and clinical trials services"),
        StockNode("ABT", "Abbott Laboratories", "Healthcare", "Medical Devices & Nutrition", "Large", "FreeStyle Libre glucose monitors, cardiac pacemakers, and Similac"),
        StockNode("DHR", "Danaher Corp", "Healthcare", "Life Sciences & Diagnostics", "Large", "Bioprocessing filtration (Cytiva), diagnostics (Beckman Coulter), and life sciences"),
        StockNode("BMY", "Bristol Myers Squibb", "Healthcare", "Hematology & Oncology", "Large", "Eliquis, Opdivo, Revlimid, and next-gen cell therapy therapies"),
        StockNode("CVS", "CVS Health", "Healthcare", "Pharmacy & Insurance", "Large", "Retail pharmacies, Caremark pharmacy benefit manager, and Aetna insurance"),
        StockNode("ELV", "Elevance Health", "Healthcare", "Health Benefits", "Large", "Anthem Blue Cross Blue Shield health insurance and Carelon services"),
        StockNode("MDT", "Medtronic plc", "Healthcare", "Medical Technology", "Large", "Cardiac rhythm management, neurosurgery, and surgical robotics"),

        # --- FINANCIALS & ASSET MANAGERS ---
        StockNode("JPM", "JPMorgan Chase", "Financials", "Global Diversified Banking", "Mega", "Largest US bank by assets, investment banking, and asset management"),
        StockNode("BAC", "Bank of America", "Financials", "Commercial & Consumer Banking", "Large", "Consumer banking branch network, Merrill wealth, and trading"),
        StockNode("WFC", "Wells Fargo", "Financials", "Consumer & Commercial Banking", "Large", "Retail banking, commercial lending, and US residential mortgages"),
        StockNode("C", "Citigroup Inc", "Financials", "Institutional & Global Banking", "Large", "Cross-border payments, treasury services, and corporate banking"),
        StockNode("GS", "Goldman Sachs Group", "Financials", "Investment Banking & Trading", "Large", "M&A advisory, equity underwriting, FICC trading, and asset management"),
        StockNode("MS", "Morgan Stanley", "Financials", "Wealth Management & Banking", "Large", "Retail wealth management (E*TRADE) and institutional securities"),
        StockNode("BRK.B", "Berkshire Hathaway", "Financials", "Conglomerate & Insurance", "Mega", "GEICO insurance, BNSF Railway, utility assets, and equity portfolio"),
        StockNode("V", "Visa Inc", "Financials", "Payment Processing Network", "Mega", "World's largest retail electronic payments processing network (VisaNet)"),
        StockNode("MA", "Mastercard Inc", "Financials", "Payment Processing Network", "Mega", "Global payment rails, cyber intelligence, and processing solutions"),
        StockNode("AXP", "American Express", "Financials", "Consumer & Corporate Cards", "Large", "Integrated payments network and premium consumer credit cards"),
        StockNode("PYPL", "PayPal Holdings", "Financials", "Digital Payments & Wallets", "Large", "PayPal digital checkout, Venmo peer-to-peer, and Braintree processing"),
        StockNode("BLK", "BlackRock Inc", "Financials", "Asset Management & Aladdin", "Large", "World's largest asset manager (iShares ETFs) and Aladdin risk software"),
        StockNode("SCHW", "Charles Schwab", "Financials", "Brokerage & Custody", "Large", "Retail brokerage accounts, TD Ameritrade integration, and RIA custody"),
        StockNode("CB", "Chubb Ltd", "Financials", "Property & Casualty Insurance", "Large", "Commercial property, casualty, personal insurance, and reinsurance"),
        StockNode("PGR", "Progressive Corp", "Financials", "Auto & Property Insurance", "Large", "Personal auto insurance with telematics (Snapshot) and commercial lines"),
        StockNode("COIN", "Coinbase Global", "Financials", "Crypto Exchange & Custody", "Large", "Retail and institutional crypto trading platform and USDC co-issuer"),
        StockNode("MSTR", "MicroStrategy Inc", "Technology", "Bitcoin Treasury & Analytics", "Large", "Institutional Bitcoin treasury company and enterprise BI software"),

        # --- ENERGY & OIL MAJORS ---
        StockNode("XOM", "Exxon Mobil Corp", "Energy", "Integrated Oil & Gas", "Mega", "Upstream exploration, Permian basin assets, refining, and chemicals"),
        StockNode("CVX", "Chevron Corp", "Energy", "Integrated Oil & Gas", "Large", "Permian production, LNG export facilities, and international refining"),
        StockNode("COP", "ConocoPhillips", "Energy", "Exploration & Production", "Large", "Pure-play oil and gas exploration in Alaska, Permian, and LNG"),
        StockNode("SLB", "Schlumberger Ltd", "Energy", "Oilfield Services & Digital", "Large", "Drilling equipment, reservoir characterization, and subsea systems"),
        StockNode("BKR", "Baker Hughes", "Energy", "Energy Technology & Turbines", "Large", "Turbomachinery, LNG compression equipment, and oilfield services"),
        StockNode("EOG", "EOG Resources", "Energy", "Shale E&P", "Large", "Low-cost crude oil and natural gas production in US shale basins"),
        StockNode("OXY", "Occidental Petroleum", "Energy", "E&P & Direct Air Capture", "Large", "Permian basin acreage, chemicals, and 1PointFive direct air capture"),
        StockNode("MPC", "Marathon Petroleum", "Energy", "Petroleum Refining", "Large", "Largest US petroleum refining system and midstream MPLX ownership"),
        StockNode("VLO", "Valero Energy", "Energy", "Refining & Renewable Fuels", "Large", "Independent petroleum refiner and renewable diesel producer"),
        StockNode("KMI", "Kinder Morgan", "Energy", "Midstream Pipelines", "Large", "North American natural gas pipeline networks and storage terminals"),
        StockNode("WMB", "The Williams Companies", "Energy", "Natural Gas Transmission", "Large", "Transco pipeline delivering natural gas to eastern US demand centers"),
        StockNode("EQT", "EQT Corp", "Energy", "Natural Gas E&P", "Large", "Largest pure-play natural gas producer in the Appalachian basin"),
        StockNode("CCJ", "Cameco Corp", "Energy", "Uranium Mining & Fuel", "Large", "High-grade uranium fuel supply and Westinghouse nuclear reactor stake"),
        StockNode("FSLR", "First Solar", "Energy", "Solar Photovoltaics", "Large", "Cadmium telluride thin-film solar modules for utility-scale solar"),

        # --- INDUSTRIALS, AEROSPACE & INFRASTRUCTURE ---
        StockNode("CAT", "Caterpillar Inc", "Industrials", "Construction & Mining Machinery", "Large", "Earthmoving machinery, mining equipment, diesel engines, and gas turbines"),
        StockNode("GE", "GE Aerospace", "Industrials", "Commercial & Military Aviation", "Large", "Commercial jet engines (LEAP CFM joint venture) and defense propulsion"),
        StockNode("LMT", "Lockheed Martin", "Industrials", "Aerospace & Defense", "Large", "F-35 Joint Strike Fighter, missile defense, and hypersonic systems"),
        StockNode("RTX", "RTX Corp", "Industrials", "Aerospace & Defense Systems", "Large", "Pratt & Whitney jet engines, Collins Aerospace, and Raytheon missiles"),
        StockNode("BA", "The Boeing Co", "Industrials", "Aerospace & Commercial Jets", "Large", "737 MAX, 787 Dreamliner commercial airliners and defense platforms"),
        StockNode("DE", "Deere & Co", "Industrials", "Agricultural Machinery", "Large", "Precision agriculture tractors, combines, and autonomous farming gear"),
        StockNode("UNP", "Union Pacific", "Industrials", "Railroad Transportation", "Large", "Transcontinental freight rail connecting western US ports and hubs"),
        StockNode("CSX", "CSX Corp", "Industrials", "Railroad Transportation", "Large", "Eastern US rail network transporting intermodal, coal, and chemicals"),
        StockNode("ODFL", "Old Dominion Freight", "Industrials", "LTL Freight Shipping", "Large", "Less-than-truckload motor carrier operating across North America"),
        StockNode("UPS", "United Parcel Service", "Industrials", "Package Delivery & Logistics", "Large", "Global courier express delivery and supply chain freight management"),
        StockNode("FDX", "FedEx Corp", "Industrials", "Express Transportation", "Large", "FedEx Express air fleet, FedEx Ground delivery, and freight shipping"),
        StockNode("HON", "Honeywell International", "Industrials", "Industrial Automation & Aero", "Large", "Aerospace avionics, building technologies, and industrial automation"),
        StockNode("PCAR", "PACCAR Inc", "Industrials", "Commercial Trucks", "Large", "Kenworth, Peterbilt, and DAF heavy-duty commercial truck manufacturing"),
        StockNode("CTAS", "Cintas Corp", "Industrials", "Uniforms & Facility Services", "Large", "Corporate uniform rental, facility services, and first-aid supplies"),
        StockNode("FAST", "Fastenal Co", "Industrials", "Industrial Supplies Distribution", "Large", "Industrial fasteners, tools, safety supplies, and vending solutions"),
        StockNode("AXON", "Axon Enterprise", "Industrials", "Public Safety Technology", "Large", "TASER conducted energy weapons, body-worn cameras, and Evidence.com"),
        StockNode("VRSK", "Verisk Analytics", "Industrials", "Risk Data & Analytics", "Large", "Data analytics and predictive risk modeling for insurance underwriting"),
        StockNode("ADP", "Automatic Data Processing", "Industrials", "Payroll & HR Solutions", "Large", "Cloud-based human capital management, payroll processing, and benefits"),
        StockNode("PAYX", "Paychex Inc", "Industrials", "Payroll & Benefits Services", "Large", "Integrated payroll, human resource, and retirement services for SMBs"),
        StockNode("NOC", "Northrop Grumman", "Industrials", "Aerospace & Defense Systems", "Large", "B-21 Raider stealth bomber, space systems, and defense electronics"),
        StockNode("GD", "General Dynamics", "Industrials", "Defense & Gulfstream Aerospace", "Large", "Virginia-class nuclear submarines, combat vehicles, and business jets"),
        StockNode("EMR", "Emerson Electric", "Industrials", "Process Automation", "Large", "Measurement instrumentation, control valves, and process automation software"),
        StockNode("ETN", "Eaton Corp", "Industrials", "Electrical Power Management", "Large", "Datacenter power distribution units, switchgear, and grid electrification"),
        StockNode("VRT", "Vertiv Holdings", "Industrials", "Datacenter Thermal & Power", "Large", "Liquid cooling systems, thermal management, and uninterrupted power (UPS)"),

        # --- UTILITIES & POWER PRODUCERS ---
        StockNode("NEE", "NextEra Energy", "Utilities", "Clean Energy & Regulated Utility", "Large", "Florida Power & Light utility and world's largest producer of solar/wind"),
        StockNode("SO", "The Southern Co", "Utilities", "Regulated Electric Utility", "Large", "Electric utility operating across the Southeast with new Vogtle nuclear units"),
        StockNode("DUK", "Duke Energy", "Utilities", "Regulated Electric Utility", "Large", "Regulated utility serving Carolinas, Florida, and Midwest customers"),
        StockNode("CEG", "Constellation Energy", "Utilities", "Clean & Nuclear Power", "Large", "America's largest producer of zero-carbon nuclear energy (Crane PPA)"),
        StockNode("VST", "Vistra Corp", "Utilities", "Merchant Power Generation", "Large", "Nuclear (Comanche Peak), natural gas generation, and energy storage"),
        StockNode("AEP", "American Electric Power", "Utilities", "Electric Utility & Transmission", "Large", "Electric generation and transmission lines serving 11 US states"),
        StockNode("EXC", "Exelon Corp", "Utilities", "Transmission & Distribution", "Large", "Pure-play transmission and distribution utility serving Chicago, PA, MD"),
        StockNode("XEL", "Xcel Energy", "Utilities", "Electric & Gas Utility", "Large", "Regulated utility leading clean energy transition across 8 Western/Midwest states"),
        StockNode("SRE", "Sempra", "Utilities", "Energy Infrastructure & LNG", "Large", "Southern California Gas, San Diego Gas & Electric, and Sempra Infrastructure"),
        StockNode("D", "Dominion Energy", "Utilities", "Electric Utility", "Large", "Virginia electric utility powering the world's largest datacenter alley"),

        # --- REAL ESTATE (REITS) ---
        StockNode("EQIX", "Equinix Inc", "Real Estate", "Datacenter REIT", "Large", "Global interconnected colocation datacenters and IBX exchanges"),
        StockNode("DLR", "Digital Realty Trust", "Real Estate", "Datacenter REIT", "Large", "Hyperscale datacenter real estate and colocation facilities worldwide"),
        StockNode("PLD", "Prologis Inc", "Real Estate", "Industrial Logistics REIT", "Large", "Modern industrial warehouse and distribution logistics centers"),
        StockNode("AMT", "American Tower", "Real Estate", "Communications Infrastructure REIT", "Large", "Multitenant communications real estate and cell phone towers"),
        StockNode("CCI", "Crown Castle", "Real Estate", "Shared Infrastructure REIT", "Large", "Cell towers, small cell nodes, and fiber optic cable miles in US"),
        StockNode("SPG", "Simon Property Group", "Real Estate", "Retail Malls REIT", "Large", "Premier shopping, dining, and mixed-use destination properties"),

        # --- MATERIALS & MINING ---
        StockNode("LIN", "Linde plc", "Materials", "Industrial Gases", "Large", "Global industrial gases supplier (oxygen, nitrogen, clean hydrogen)"),
        StockNode("SHW", "Sherwin-Williams", "Materials", "Paints & Coatings", "Large", "Architectural paints, industrial coatings, and automotive finishes"),
        StockNode("FCX", "Freeport-McMoRan", "Materials", "Copper & Gold Mining", "Large", "Leading international mining company producing copper for AI grid & EVs"),
        StockNode("NEM", "Newmont Corp", "Materials", "Gold Mining", "Large", "World's leading gold mining company with copper/silver/zinc production"),
        StockNode("NUE", "Nucor Corp", "Materials", "Steel Minimills", "Large", "Largest steel and steel products manufacturer in North America via scrap EAF"),

        # --- BROAD MARKET INDEXES & SECTOR BENCHMARKS ---
        StockNode("SPY", "SPDR S&P 500 ETF", "Index", "Broad Market US Large-Cap", "Mega", "Benchmark tracking the 500 largest publicly traded US companies"),
        StockNode("QQQ", "Invesco QQQ Trust", "Index", "Large-Cap Tech Benchmark", "Mega", "Benchmark tracking the 100 largest non-financial NASDAQ companies"),
        StockNode("DIA", "SPDR Dow Jones Industrial", "Index", "Blue-Chip US Equities", "Mega", "Benchmark tracking 30 prominent blue-chip US industrial leaders"),
        StockNode("IWM", "iShares Russell 2000 ETF", "Index", "US Small-Cap Benchmark", "Mega", "Benchmark tracking the US small-cap equity universe"),
        StockNode("SMH", "VanEck Semiconductor ETF", "Index", "Semiconductor Sector", "Mega", "Benchmark tracking global leaders in semiconductor design and manufacturing"),
        StockNode("XLF", "Financial Select Sector SPDR", "Index", "Financial Sector", "Mega", "Benchmark tracking S&P 500 financial institutions and payment rails"),
        StockNode("XLE", "Energy Select Sector SPDR", "Index", "Energy Sector", "Mega", "Benchmark tracking S&P 500 oil, gas, and energy infrastructure"),
        StockNode("XLV", "Health Care Select Sector SPDR", "Index", "Healthcare Sector", "Mega", "Benchmark tracking S&P 500 pharma, biotech, and medical devices"),
        StockNode("XLI", "Industrial Select Sector SPDR", "Index", "Industrial Sector", "Mega", "Benchmark tracking S&P 500 aerospace, defense, machinery, and rail"),
        StockNode("XLU", "Utilities Select Sector SPDR", "Index", "Utilities Sector", "Mega", "Benchmark tracking S&P 500 electric, gas, and nuclear power utilities"),
        StockNode("XLK", "Technology Select Sector SPDR", "Index", "Technology Sector", "Mega", "Benchmark tracking S&P 500 enterprise software, semis, and hardware"),
        StockNode("XLP", "Consumer Staples Sector SPDR", "Index", "Staples Sector", "Mega", "Benchmark tracking S&P 500 food, beverage, and household essentials"),
        StockNode("XLY", "Consumer Discretionary SPDR", "Index", "Discretionary Sector", "Mega", "Benchmark tracking S&P 500 retail, travel, apparel, and autos"),
    ]


def get_market_edges() -> List[GraphEdge]:
    """Return all curated GraphEdges connecting constituents across supply chains and competitor rings."""
    edges: List[GraphEdge] = []

    def c(src: str, tgt: str, w: float = 0.9, desc: str = ""):
        """Bidirectional competitor or peer edge."""
        edges.append(GraphEdge(src, tgt, RelationType.COMPETITOR.value, w, desc, bidirectional=True))

    def p(src: str, tgt: str, w: float = 0.85, desc: str = ""):
        """Bidirectional correlated peer edge."""
        edges.append(GraphEdge(src, tgt, RelationType.CORRELATED_PEER.value, w, desc, bidirectional=True))

    def s(src: str, tgt: str, w: float = 0.9, desc: str = ""):
        """Directed supplier -> customer edge."""
        edges.append(GraphEdge(src, tgt, RelationType.SUPPLIER_TO.value, w, desc))

    def infra(src: str, tgt: str, w: float = 0.9, desc: str = ""):
        """Directed infrastructure partner edge."""
        edges.append(GraphEdge(src, tgt, RelationType.INFRASTRUCTURE_PARTNER.value, w, desc))

    def pwr(src: str, tgt: str, w: float = 0.9, desc: str = ""):
        """Directed power supplier -> datacenter buyer edge."""
        edges.append(GraphEdge(src, tgt, RelationType.POWER_PARTNER.value, w, desc))

    # =========================================================================
    # 1. SEMICONDUCTOR MANUFACTURING, EQUIPMENT & SUPPLY CHAIN
    # =========================================================================
    # Lithography and equipment to foundries
    s("ASML", "TSM", 1.0, "Sole supplier of High-NA and EUV lithography tools")
    s("ASML", "INTC", 0.9, "Supplier of EUV scanners for Intel 18A process")
    s("ASML", "MU", 0.85, "EUV scanners for advanced 1-gamma DRAM and HBM")
    s("AMAT", "TSM", 0.95, "Key supplier of deposition, etch, and CMP systems")
    s("AMAT", "INTC", 0.9, "Materials engineering equipment for US foundries")
    s("AMAT", "MU", 0.9, "Deposition equipment for high-aspect ratio memory")
    s("LRCX", "TSM", 0.9, "Atomic-layer etching and deposition systems")
    s("LRCX", "MU", 0.95, "Primary dry etch systems for 3D NAND and HBM memory")
    s("LRCX", "INTC", 0.85, "Dielectric etch tools for leading-edge logic")
    s("KLAC", "TSM", 0.9, "Yield inspection and optical metrology systems")
    s("KLAC", "INTC", 0.85, "Defect review and process control tools")
    s("KLAC", "MU", 0.85, "Yield management for high-density DRAM packaging")

    # Equipment peer ring
    c("ASML", "AMAT", 0.8, "Competition in fab tool spend and wafer engineering")
    c("AMAT", "LRCX", 0.9, "Rivalry in dry etch, deposition, and materials engineering")
    c("LRCX", "KLAC", 0.8, "Competition for fab tool budgets and yield solutions")

    # Industrial gases for fabs
    s("LIN", "TSM", 0.9, "Ultra-high purity industrial gases for advanced node fabs")
    s("LIN", "INTC", 0.9, "Specialty gases and clean hydrogen for US/Europe foundries")

    # EDA software to chip designers
    s("SNPS", "NVDA", 0.95, "EDA software, synthesis, and digital IP blocks")
    s("SNPS", "AMD", 0.9, "Silicon design verification tools")
    s("SNPS", "AAPL", 0.9, "EDA tools for custom Apple Silicon processors")
    s("SNPS", "INTC", 0.85, "EDA tools for Intel foundry and design teams")
    s("CDNS", "NVDA", 0.9, "Simulation and hardware emulation systems")
    s("CDNS", "QCOM", 0.9, "Electronic design automation tools for Snapdragon")
    s("CDNS", "AAPL", 0.85, "Custom silicon simulation and verification software")
    s("ANSS", "NVDA", 0.85, "Multiphysics thermal and electromagnetic simulation for GPUs")
    s("ANSS", "AMD", 0.85, "Chiplet interconnect simulation and thermal analysis")
    c("SNPS", "CDNS", 0.95, "Duopoly rivalry across electronic design automation (EDA)")
    c("CDNS", "ANSS", 0.85, "Simulation and design automation competition")
    c("ADSK", "ANSS", 0.85, "Engineering design vs multiphysics simulation software")
    c("ADSK", "CDNS", 0.8, "CAD architecture vs electronic design automation software")

    # Foundry to fabless chip designers
    s("TSM", "NVDA", 1.0, "Exclusive foundry manufacturing for Blackwell, Hopper, and Rubin AI GPUs")
    s("TSM", "AMD", 0.95, "Primary foundry for EPYC processors and Instinct MI300/325 GPUs")
    s("TSM", "AAPL", 1.0, "Sole manufacturer of A-series and M-series custom silicon")
    s("TSM", "QCOM", 0.9, "Advanced node wafer fabrication for Snapdragon mobile/PC platforms")
    s("TSM", "AVGO", 0.9, "Manufacturing partner for custom AI ASIC accelerators and switches")
    s("TSM", "MRVL", 0.85, "Foundry for cloud-optimized custom compute silicon")

    # Memory to AI accelerators
    s("MU", "NVDA", 0.95, "Key supplier of HBM3e high-bandwidth memory for H200 and Blackwell")
    s("MU", "AMD", 0.9, "HBM3e supplier for MI300/325 series accelerators")

    # Silicon architecture IP licensing
    s("ARM", "AAPL", 0.95, "Instruction set architecture licensing for all Apple devices")
    s("ARM", "QCOM", 0.9, "Architectural foundations for mobile and Windows on Snapdragon chips")
    s("ARM", "NVDA", 0.85, "Architecture for Grace CPU and superchip systems")
    s("ARM", "AMZN", 0.85, "Architecture IP for AWS Graviton cloud server processors")
    s("ARM", "GOOGL", 0.8, "Custom silicon CPU architecture for Google Axion processors")

    # Chipmaker competitor rings
    c("NVDA", "AMD", 0.95, "Direct rivalry in datacenter AI GPUs, CUDA vs ROCm, and gaming graphics")
    c("INTC", "AMD", 0.95, "x86 server (Xeon vs EPYC) and PC desktop CPU competition")
    c("NVDA", "INTC", 0.85, "Competition between discrete GPUs and server CPUs")
    c("AVGO", "NVDA", 0.8, "Competition between custom hyperscaler ASICs and general GPUs")
    c("QCOM", "ARM", 0.75, "Oryon architecture licensing dispute and PC CPU competition")
    c("QCOM", "AVGO", 0.8, "Competition in mobile wireless, connectivity, and RF front-ends")
    c("MRVL", "AVGO", 0.9, "Direct competition in custom datacenter ASICs and optical DSPs")
    c("TXN", "ADI", 0.95, "Head-to-head rivalry in analog, power management, and converters")
    c("TXN", "NXPI", 0.85, "Industrial and automotive analog processor competition")
    c("NXPI", "ON", 0.85, "Automotive electrification and microcontroller competition")
    c("ON", "MCHP", 0.85, "Embedded power and microcontroller market competition")
    c("MCHP", "TXN", 0.85, "Embedded microcontrollers and mixed-signal components")

    # =========================================================================
    # 2. AI SERVERS, SYSTEMS & DATACENTER HARDWARE
    # =========================================================================
    # GPU suppliers to hardware builders
    s("NVDA", "SMCI", 0.95, "Supplies Blackwell/Hopper GPUs for liquid-cooled rack clusters")
    s("NVDA", "DELL", 0.95, "Supplies AI accelerator cards for Dell PowerEdge servers")
    s("NVDA", "HPE", 0.9, "Supplies GPUs for HPE Cray supercomputers and ProLiant systems")
    s("AMD", "DELL", 0.85, "Supplies EPYC CPUs and Instinct GPUs for Dell servers")
    s("AMD", "HPE", 0.85, "Supplies EPYC processors for frontier supercomputers")
    c("SMCI", "DELL", 0.9, "Rivalry in AI server rack integration and enterprise deployment")
    c("DELL", "HPE", 0.95, "Enterprise server, storage, and hybrid cloud infrastructure rivalry")
    c("DELL", "HPQ", 0.85, "Commercial and consumer PC hardware rivalry")
    s("INTC", "DELL", 0.9, "Core and Xeon CPUs for Dell enterprise PCs and servers")
    s("INTC", "HPQ", 0.9, "Processors for HP consumer and commercial laptop fleet")

    # Networking infrastructure
    c("CSCO", "HPE", 0.85, "Enterprise networking and campus switching (Cisco vs Aruba)")
    c("CSCO", "PANW", 0.85, "Enterprise network security and firewall hardware competition")
    s("CSCO", "MSFT", 0.85, "Enterprise datacenter switching and optical routing")
    s("CSCO", "AMZN", 0.85, "AWS datacenter network routing infrastructure")
    c("CSCO", "MSFT", 0.8, "Enterprise collaboration tools: Webex vs Microsoft Teams")

    # Datacenter cooling, power distribution and facilities
    infra("VRT", "NVDA", 0.95, "Co-development of liquid cooling solutions for GB200 NVL72 racks")
    infra("VRT", "MSFT", 0.9, "Thermal management and uninterrupted power systems for Azure")
    infra("VRT", "AMZN", 0.9, "Cooling systems for AWS global datacenter fleet")
    infra("VRT", "META", 0.85, "Liquid-to-air cooling architectures for Meta AI clusters")
    infra("VRT", "ORCL", 0.85, "Thermal management for high-density OCI AI clusters")
    infra("ETN", "MSFT", 0.9, "Electrical switchgear and power distribution for datacenters")
    infra("ETN", "AMZN", 0.9, "Datacenter power management and grid infrastructure")
    infra("ETN", "GOOGL", 0.85, "Substations and electrical systems for Google Cloud")
    c("VRT", "ETN", 0.85, "Competition in datacenter electrical power and thermal systems")
    s("FCX", "ETN", 0.85, "Copper supply for electrical equipment and power grid transformers")

    # Datacenter colocation and REITs
    infra("EQIX", "MSFT", 0.9, "Interconnection and direct cloud on-ramps to Azure")
    infra("EQIX", "AMZN", 0.9, "Direct Connect colocation facilities for AWS")
    infra("EQIX", "ORCL", 0.85, "Interconnection and private on-ramps for Oracle Cloud")
    infra("DLR", "MSFT", 0.9, "Hyperscale datacenter leasing and cloud campus hosting")
    infra("DLR", "GOOGL", 0.85, "Turnkey datacenter facilities for Google Cloud")
    infra("DLR", "META", 0.85, "Leased hyperscale datacenter space for Meta AI")
    c("EQIX", "DLR", 0.95, "Global datacenter REIT rivalry in colocation and interconnection")
    c("AMT", "CCI", 0.9, "Cell tower and wireless infrastructure REIT rivalry")
    s("AMT", "TMUS", 0.85, "Leases cellular tower space for T-Mobile 5G coverage")
    s("CCI", "TMUS", 0.85, "Provides cell towers, small cells, and fiber backhaul")
    c("PLD", "EQIX", 0.75, "Specialized infrastructure REIT investment basket")

    # =========================================================================
    # 3. HYPERSCALERS & ENTERPRISE CLOUD ECOSYSTEM
    # =========================================================================
    # Nvidia AI chips to Hyperscalers
    infra("NVDA", "MSFT", 1.0, "Massive deployment of DGX/GPU clusters on Azure cloud")
    infra("NVDA", "AMZN", 0.95, "Primary accelerator provider for AWS EC2 GPU instances")
    infra("NVDA", "GOOGL", 0.9, "Supplies GPUs alongside Google's internal TPU deployment")
    infra("NVDA", "META", 0.95, "Meta is one of Nvidia's largest cluster buyers for Llama training")
    infra("NVDA", "ORCL", 0.9, "Powering Oracle Cloud Infrastructure (OCI) superclusters")

    # Hyperscaler competitor mesh
    c("MSFT", "GOOGL", 0.95, "Intense rivalry across cloud (Azure/GCP), search, and AI models")
    c("MSFT", "AMZN", 0.95, "Head-to-head battle between Azure and AWS cloud platforms")
    c("GOOGL", "AMZN", 0.9, "Cloud computing, retail advertising, and digital assistant rivalry")
    c("META", "GOOGL", 0.95, "Digital advertising dominance and open-source vs closed AI race")
    c("AAPL", "GOOGL", 0.9, "iOS vs Android mobile operating systems and app store ecosystem")
    p("GOOGL", "GOOG", 1.0, "Alphabet Class A and Class C dual equity shares")

    # Enterprise SaaS & Cloud Data partnerships
    infra("PLTR", "MSFT", 0.9, "Strategic alliance deploying AIP with Azure OpenAI inside defense")
    infra("PLTR", "AMZN", 0.85, "Deployment of Palantir Foundry and Gotham on AWS GovCloud")
    infra("SNOW", "AMZN", 0.9, "Primary cloud data warehousing partner on AWS")
    infra("SNOW", "MSFT", 0.85, "Snowflake on Azure enterprise data integration")
    infra("MDB", "AMZN", 0.85, "MongoDB Atlas native deployment and co-selling on AWS")
    infra("MDB", "MSFT", 0.85, "MongoDB Atlas enterprise integration on Microsoft Azure")
    infra("MDB", "GOOGL", 0.8, "MongoDB Atlas integration on Google Cloud Platform")
    infra("DDOG", "AMZN", 0.9, "Deep integration for AWS cloud monitoring and security")
    infra("DDOG", "MSFT", 0.85, "Azure native integration for enterprise observability")
    infra("DDOG", "GOOGL", 0.8, "Observability and monitoring for Google Cloud workloads")
    infra("ORCL", "MSFT", 0.9, "Oracle Database@Azure multi-cloud strategic integration")
    infra("ORCL", "GOOGL", 0.85, "Oracle Database@GoogleCloud partnership")
    infra("NOW", "MSFT", 0.85, "ServiceNow workflow orchestration integrated with Microsoft 365")
    infra("CRM", "AMZN", 0.85, "Salesforce core infrastructure hosted on AWS")
    infra("CRM", "MSFT", 0.8, "Salesforce integration with Microsoft Teams and Windows")
    infra("CRWD", "AMZN", 0.85, "Falcon cybersecurity native protection for AWS workloads")
    infra("CRWD", "MSFT", 0.8, "Protects enterprise Windows and Azure endpoints")
    infra("PANW", "MSFT", 0.8, "Prisma Cloud security integration with Azure")
    infra("ZS", "MSFT", 0.85, "Zero Trust Exchange integration for Microsoft 365 and Azure")

    # Enterprise Software competitor rings
    c("CRM", "NOW", 0.9, "Enterprise workflow and customer service platform rivalry")
    c("CRM", "MSFT", 0.85, "Salesforce vs Microsoft Dynamics 365 CRM competition")
    c("NOW", "WDAY", 0.85, "Enterprise HR and digital workflow orchestration competition")
    c("NOW", "TEAM", 0.8, "ServiceNow IT Service Management vs Jira Service Management")
    c("SNOW", "MDB", 0.85, "Data warehouse vs operational document database competition")
    c("SNOW", "DDOG", 0.8, "Cloud data analytics vs observability platform overlap")
    c("PANW", "CRWD", 0.95, "Fierce rivalry in next-generation cybersecurity and endpoint SASE")
    c("CRWD", "FTNT", 0.85, "Endpoint cloud security vs network firewall hardware competition")
    c("PANW", "FTNT", 0.9, "Enterprise firewall and secure access service edge (SASE) rivalry")
    c("CRWD", "ZS", 0.85, "Zero trust cloud security and endpoint protection integration/overlap")
    c("ZS", "PANW", 0.9, "Zero Trust SASE and cloud network security rivalry")
    c("INTU", "ADP", 0.8, "QuickBooks payroll vs ADP small-business payroll services")
    c("ADP", "PAYX", 0.95, "Direct rivalry in enterprise and SMB payroll processing")
    c("WDAY", "ADP", 0.85, "Workday enterprise human capital management vs ADP payroll")
    c("ADBE", "CRM", 0.8, "Adobe Experience Cloud vs Salesforce Marketing Cloud")
    c("ADBE", "APP", 0.75, "Digital creative software vs mobile ad monetisation tech")
    c("APP", "TTD", 0.85, "Programmatic ad networks and app monetization technology")
    c("TTD", "GOOGL", 0.85, "Independent DSP vs Google Display & Video 360")
    c("TTD", "META", 0.8, "Open internet advertising vs Meta walled garden")
    c("TEAM", "MSFT", 0.8, "Jira and Confluence vs Microsoft GitHub, Azure DevOps, and Teams")
    c("CDW", "CTSH", 0.8, "IT hardware/software procurement vs technology consulting")
    c("CTSH", "ACN", 0.85, "Global IT services and digital transformation consulting rivalry")
    c("IBM", "ACN", 0.85, "Enterprise technology consulting and cloud migration competition")
    c("IBM", "ORCL", 0.8, "Legacy enterprise relational database and hybrid cloud overlap")
    c("ROP", "VRSK", 0.85, "Vertical market software solutions and analytical data assets")
    c("ROP", "NOW", 0.75, "Niche enterprise workflow software peers")
    s("CDW", "MSFT", 0.9, "Premier Microsoft Cloud Solution Provider distributing licenses")
    s("CDW", "CSCO", 0.9, "Cisco Gold Certified Partner distributing networking hardware")

    # Law enforcement and vertical data tech
    infra("AXON", "MSFT", 0.85, "Azure Government cloud hosting for Evidence.com video data")
    infra("AXON", "AMZN", 0.8, "AWS GovCloud secure storage for body camera feeds")
    c("AXON", "LMT", 0.75, "Public safety technology vs defense prime contractor")

    # =========================================================================
    # 4. DATACENTER POWER, NUCLEAR & UTILITIES
    # =========================================================================
    pwr("CEG", "MSFT", 0.95, "20-year power purchase agreement to restart Three Mile Island nuclear plant")
    pwr("VST", "META", 0.9, "Power supply agreements and clean energy contracts in ERCOT Texas")
    pwr("VST", "AMZN", 0.85, "Comanche Peak nuclear and merchant power for AWS datacenters")
    pwr("NEE", "GOOGL", 0.9, "Clean wind and solar generation contracts powering Google datacenters")
    pwr("D", "AMZN", 0.95, "Electric utility supplying Northern Virginia Data Center Alley")
    pwr("D", "MSFT", 0.9, "Power delivery for massive Azure datacenter buildouts in Virginia")
    s("CCJ", "CEG", 0.9, "Uranium fuel supply for Constellation's commercial nuclear reactor fleet")
    s("CCJ", "VST", 0.85, "Uranium supply contracts for Comanche Peak nuclear generation")
    s("KMI", "VST", 0.85, "Natural gas pipeline transmission fuel for power plants")
    s("WMB", "CEG", 0.8, "Natural gas delivery infrastructure for merchant peaking stations")

    # Utility competitor/peer ring
    c("CEG", "VST", 0.95, "Merchant power producers benefiting from AI datacenter load growth")
    p("NEE", "SO", 0.9, "Major southeastern regulated electric utility peers")
    p("SO", "DUK", 0.9, "Regulated southeastern utility peers with growing datacenter demand")
    p("DUK", "AEP", 0.85, "Multistate regulated electric utility transmission peers")
    p("AEP", "EXC", 0.85, "Major Midwest and Mid-Atlantic transmission utility peers")
    p("EXC", "XEL", 0.85, "Regulated transmission and clean energy transition peers")
    p("XEL", "NEE", 0.8, "Western and Midwest clean wind/solar utility peers")
    p("SRE", "NEE", 0.8, "Regulated energy infrastructure and renewable developers")
    p("SRE", "SO", 0.8, "Regulated gas and electric infrastructure peers")
    p("D", "SO", 0.85, "Southeastern and Mid-Atlantic utility peers")
    p("CEG", "NEE", 0.85, "Zero-carbon clean power generation leaders")
    s("FSLR", "NEE", 0.85, "Solar module supplier for NextEra utility-scale solar projects")
    s("FSLR", "DUK", 0.8, "Solar panels for Duke Energy regulated decarbonization buildouts")

    # =========================================================================
    # 5. FINANCIALS, MEGABANKS & PAYMENT NETWORKS
    # =========================================================================
    # Universal Megabanks peer mesh
    c("JPM", "BAC", 0.95, "Top two US commercial banks competing in retail deposits and capital markets")
    c("JPM", "WFC", 0.9, "Major consumer banking, lending, and branch network rivalry")
    c("BAC", "WFC", 0.9, "Consumer checking, auto loans, and retail mortgage competition")
    c("JPM", "C", 0.9, "Global institutional cash management, treasury, and corporate banking")
    c("BAC", "C", 0.85, "International corporate banking and foreign exchange services")

    # Investment Banks & Brokerages
    c("GS", "MS", 0.95, "Wall Street rivalry in M&A advisory, equity underwriting, and trading")
    c("GS", "JPM", 0.9, "Investment banking league table competition")
    c("MS", "SCHW", 0.9, "Wealth management, financial advisory, and retail brokerage rivalry")
    c("BLK", "SCHW", 0.85, "Asset management, iShares vs Schwab ETF inflows")
    c("CB", "PGR", 0.85, "Commercial casualty vs personal lines property/auto insurance")
    s("VRSK", "CB", 0.9, "Insurance underwriting data and catastrophic risk modeling")
    s("VRSK", "PGR", 0.9, "Auto actuarial data and claims predictive analytics")
    s("PGR", "CPRT", 0.9, "Total loss salvage vehicle remarketing auctions")
    s("CB", "CPRT", 0.85, "Commercial casualty vehicle salvage processing")

    # Payment rails and fintech
    c("V", "MA", 0.98, "Global duopoly in electronic payment networks and card transaction routing")
    c("V", "AXP", 0.85, "Payment networks vs closed-loop premium card spend")
    c("MA", "AXP", 0.85, "Corporate travel and consumer credit card spend rivalry")
    c("PYPL", "V", 0.8, "Digital wallet checkout vs underlying VisaNet debit/credit card network")
    c("PYPL", "AXP", 0.8, "Digital wallet competition in online merchant checkout")
    s("V", "AMZN", 0.9, "Payment network processing for global e-commerce checkouts")
    s("MA", "AMZN", 0.9, "Payment rails processing for Amazon online and AWS transactions")
    s("V", "BKNG", 0.85, "Global travel booking credit and debit processing")
    s("MA", "ABNB", 0.85, "Host payout and guest travel booking payment rails")
    s("V", "WMT", 0.9, "In-store and online retail card swipe processing")
    s("V", "COST", 0.95, "Exclusive in-store credit card network acceptance for Costco US")
    s("V", "MELI", 0.85, "Mercado Pago card issuance on Visa payment network")
    s("V", "UBER", 0.9, "Instant driver earnings cash out via Visa Direct rails")
    s("V", "PYPL", 0.9, "Underlying clearing and settlement for PayPal and Venmo cards")

    # Berkshire Hathaway anchor holdings
    p("BRK.B", "AAPL", 0.9, "Berkshire Hathaway's largest public common stock investment")
    p("BRK.B", "BAC", 0.85, "Berkshire Hathaway major banking holding")
    p("BRK.B", "AXP", 0.85, "Berkshire Hathaway long-term core equity holding")
    p("BRK.B", "OXY", 0.8, "Berkshire Hathaway majority equity and preferred stake")
    p("BRK.B", "PGR", 0.8, "GEICO insurance competition against Progressive auto lines")

    # Crypto ecosystem
    p("COIN", "MSTR", 0.9, "High statistical beta to Bitcoin price action and digital asset inflows")
    infra("COIN", "BLK", 0.9, "Exclusive custodian for BlackRock iShares Bitcoin Trust (IBIT)")
    s("MSTR", "MSFT", 0.8, "MicroStrategy enterprise analytics software running on Azure")

    # =========================================================================
    # 6. HEALTHCARE, BIOTECH & PHARMACEUTICALS
    # =========================================================================
    # Obesity & Diabetes GLP-1 war
    c("LLY", "NVO", 0.98, "Global duopoly battle in GLP-1 / GIP therapies (Mounjaro/Zepbound vs Ozempic/Wegovy)")
    c("LLY", "MRK", 0.85, "Pharmaceutical mega-cap competition and oncology pipeline rivalry")
    c("LLY", "PFE", 0.85, "Large-cap pharma pipeline, patent battles, and commercial execution")
    c("MRK", "PFE", 0.9, "Vaccines and oncology drug portfolio rivalry")
    c("MRK", "BMY", 0.9, "Immuno-oncology checkpoint inhibitors (Keytruda vs Opdivo)")
    c("ABBV", "JNJ", 0.85, "Immunology and targeted oncology therapeutic rivalry")
    c("JNJ", "PFE", 0.85, "Consumer health heritage, vaccines, and global pharma operations")
    c("BMY", "ABBV", 0.85, "Hematology and oncology therapeutics competition")

    # Biotechnology peers
    c("VRTX", "REGN", 0.85, "Premier biotechs pioneering targeted genetic and antibody treatments")
    c("REGN", "AMGN", 0.85, "Monoclonal antibodies and biopharma therapeutic competition")
    c("AMGN", "GILD", 0.85, "Biotechnology oncology and immunology drug pipelines")
    c("GILD", "VRTX", 0.8, "Specialty biopharma and drug discovery pipelines")
    c("BIIB", "LLY", 0.85, "Alzheimer's disease therapies (Leqembi vs Kisunla)")
    c("BIIB", "REGN", 0.8, "Neuroscience and immunology biopharma research")
    c("MRNA", "PFE", 0.9, "mRNA vaccine technologies and post-pandemic infectious disease pipeline")
    c("MRNA", "VRTX", 0.8, "Genetic medicine and RNA-based therapeutic platforms")

    # Managed care and pharmacy
    c("UNH", "ELV", 0.95, "Commercial and Medicare health insurance plan rivalry")
    c("UNH", "CVS", 0.9, "Health insurance (UnitedHealthcare vs Aetna) and pharmacy services")
    c("CVS", "ELV", 0.85, "Pharmacy benefit management and commercial health benefits")
    s("CVS", "NVO", 0.9, "Caremark PBM formulary placement for Wegovy and Ozempic")
    s("UNH", "LLY", 0.85, "UnitedHealthcare formulary coverage for Mounjaro/Zepbound")
    s("CVS", "BMY", 0.85, "Caremark PBM distribution of Eliquis blood thinner")

    # Medical devices and life science tools
    c("ISRG", "MDT", 0.85, "Robotic surgery (da Vinci vs Hugo) and minimally invasive systems")
    c("MDT", "ABT", 0.9, "Cardiovascular, pacemakers, and medical device rivalry")
    c("DXCM", "ABT", 0.95, "Continuous glucose monitoring head-to-head (Dexcom G7 vs FreeStyle Libre)")
    c("TMO", "DHR", 0.95, "Duopoly in life sciences instrumentation, bioprocessing, and clinical reagents")
    c("DHR", "ABT", 0.85, "Clinical diagnostics and laboratory automation instruments")
    c("ILMN", "TMO", 0.85, "Genomic sequencing instruments and molecular diagnostics")
    c("IDXX", "TMO", 0.8, "Diagnostic testing instruments and consumable veterinary assays")
    c("IDXX", "ABT", 0.8, "Point-of-care veterinary diagnostics vs human diagnostic instruments")
    c("GEHC", "MDT", 0.8, "Hospital diagnostic imaging and patient surgical monitoring systems")
    c("GEHC", "ISRG", 0.8, "Surgical navigation imaging vs robotic operative instrumentation")
    p("GEHC", "GE", 0.85, "Spun off from GE; healthcare imaging legacy peer")

    # Cross-sector tech & AI to healthcare
    infra("NVDA", "LLY", 0.85, "AI drug discovery collaboration utilizing BioNeMo supercomputing")
    infra("NVDA", "ISRG", 0.85, "Real-time AI video analytics and compute for surgical robotics")
    s("UNH", "ISRG", 0.8, "Insurance coverage and hospital reimbursement for robotic procedures")
    s("UNH", "DXCM", 0.85, "Formulary insurance coverage for diabetes glucose monitors")
    s("ILMN", "REGN", 0.85, "High-throughput sequencers for the Regeneron Genetics Center")
    s("DHR", "PFE", 0.85, "Bioprocessing filtration and chromatography resins for drug production")

    # =========================================================================
    # 7. ENERGY, OIL, GAS & MINING
    # =========================================================================
    # Integrated Supermajors
    c("XOM", "CVX", 0.98, "Global integrated oil & gas rivalry across Permian, offshore, and refining")
    c("XOM", "COP", 0.9, "Permian shale exploration and LNG export competition")
    c("CVX", "COP", 0.9, "Upstream exploration, deepwater, and unconventional shale rivalry")
    c("COP", "EOG", 0.9, "US shale basin pure-play exploration & production competition")
    c("COP", "OXY", 0.85, "Permian basin acreage and crude production competition")
    c("EOG", "OXY", 0.9, "Low-cost unconventional shale drilling rivalry")

    # Oilfield services and tech
    c("SLB", "BKR", 0.95, "Global oilfield services, reservoir evaluation, and drilling technology")
    s("SLB", "XOM", 0.9, "Drilling services and digital subsea completion for Exxon")
    s("SLB", "CVX", 0.9, "Reservoir characterization and offshore oilfield equipment")
    s("BKR", "CVX", 0.85, "Turbomachinery and LNG refrigeration compressors for Chevron LNG")
    s("BKR", "XOM", 0.85, "Gas turbines and carbon capture compression systems")

    # Midstream and Refining
    c("MPC", "VLO", 0.95, "Leading US independent petroleum refining and renewable diesel rivalry")
    c("MPC", "XOM", 0.85, "Refining crack spreads and retail fuel marketing competition")
    c("VLO", "CVX", 0.85, "Gulf Coast and West Coast refining margin competition")
    c("KMI", "WMB", 0.95, "North American interstate natural gas pipeline network rivalry")
    s("KMI", "COP", 0.85, "Gathering and takeaway pipeline transport for Permian gas")
    s("WMB", "EQT", 0.85, "Appalachian basin natural gas gathering")

    # Clean Tech, Materials and Mining
    s("LIN", "XOM", 0.85, "Industrial hydrogen supply for low-carbon fuels and refinery desulfurization")
    c("FCX", "NEM", 0.8, "Global copper and gold mining industry peers")
    p("NEM", "SPY", 0.75, "Gold mining inflation hedge and materials constituent")
    s("NUE", "CAT", 0.85, "Structural steel plate and bars for construction machinery")
    s("NUE", "DE", 0.85, "High-strength steel for agricultural combine harvesters")
    s("NUE", "BA", 0.8, "Specialty steel alloys for aerospace tooling")

    # =========================================================================
    # 8. INDUSTRIALS, DEFENSE, MACHINERY & LOGISTICS
    # =========================================================================
    # Defense Prime Contractors
    c("LMT", "RTX", 0.95, "Pentagon budget rivalry in air defense, radar, and precision munitions")
    c("LMT", "BA", 0.9, "Military fighter aircraft and aerospace defense competition")
    c("LMT", "NOC", 0.9, "Stealth bombers, nuclear deterrent triad, and missile defense")
    c("RTX", "NOC", 0.85, "Defense electronics, advanced sensors, and missile systems")
    c("NOC", "GD", 0.85, "Combat systems, ordnance, and defense technology")
    c("GD", "LMT", 0.85, "Defense prime contractors for Pentagon land, sea, and air systems")
    c("GD", "BA", 0.8, "Gulfstream business jets vs commercial aerospace")

    # Commercial Aerospace
    c("GE", "RTX", 0.95, "Rivalry in commercial jet engines (GE Aerospace / CFM vs Pratt & Whitney GTF)")
    c("GE", "HON", 0.85, "Aerospace engines, auxiliary power systems, and mechanical flight controls")
    s("GE", "BA", 0.95, "Primary jet engine supplier for Boeing 777X and 737 MAX commercial airliners")
    s("RTX", "BA", 0.9, "Supplies Collins Aerospace avionics, flight controls, and interiors to Boeing")
    s("HON", "BA", 0.9, "Supplies avionics, cockpit displays, and brake systems to Boeing")

    # Machinery, Automation & Rail
    c("CAT", "DE", 0.95, "Heavy earthmoving and construction equipment rivalry")
    c("CAT", "PCAR", 0.8, "Commercial heavy-duty truck powertrains and diesel engines")
    c("DE", "PCAR", 0.8, "Heavy diesel engine design and commercial vehicle manufacturing")
    c("HON", "EMR", 0.9, "Industrial process control, building automation, and sensors")
    c("EMR", "ETN", 0.85, "Electrical automation and industrial grid power management")
    s("EMR", "XOM", 0.85, "Process automation and instrumentation for chemical refineries")
    s("ADSK", "CAT", 0.85, "3D CAD modeling and digital twin software for equipment manufacturing")
    c("UNP", "CSX", 0.95, "Class I freight railroads connecting Western and Eastern US corridors")
    c("ODFL", "CSX", 0.8, "Less-than-truckload trucking vs rail freight transportation")
    c("ODFL", "UPS", 0.85, "Freight shipping and freight logistics competition")
    s("PCAR", "UNP", 0.8, "Intermodal transportation of commercial freight trucks")
    c("FAST", "CTAS", 0.85, "Industrial plant supplies, maintenance, and facility services")
    s("FAST", "GE", 0.85, "Industrial fasteners and MRO supplies for aerospace manufacturing")
    s("FAST", "BA", 0.85, "Fasteners and safety tooling for commercial aircraft assembly")
    s("CTAS", "CAT", 0.85, "Uniform and protective gear services for machinery plants")
    s("CTAS", "WMT", 0.85, "Facility services and cleaning solutions for retail stores")

    # Logistics & Delivery to E-Commerce
    c("UPS", "FDX", 0.98, "Global parcel logistics and air express delivery duopoly")
    s("UPS", "AMZN", 0.9, "Major parcel shipping and last-mile delivery partner for Amazon")
    s("FDX", "WMT", 0.85, "Express package and e-commerce fulfillment partner for Walmart")
    s("FDX", "TGT", 0.85, "Parcel delivery and fulfillment services for Target online orders")
    s("PLD", "AMZN", 0.95, "Largest landlord leasing modern distribution logistics warehouses to Amazon")
    s("PLD", "WMT", 0.9, "Logistics warehouse leasing for Walmart fulfillment centers")
    s("PLD", "HD", 0.85, "Fulfillment distribution centers for The Home Depot")

    # =========================================================================
    # 9. CONSUMER RETAIL, DISCRETIONARY, STAPLES & MEDIA
    # =========================================================================
    # Mega Retailers
    c("WMT", "TGT", 0.95, "Head-to-head discount retail and grocery rivalry")
    c("WMT", "COST", 0.9, "Retail dominance, bulk pricing, and grocery market share")
    c("WMT", "AMZN", 0.95, "Omnichannel retail vs e-commerce leader")
    c("TGT", "COST", 0.85, "Consumer merchandise and suburban retail competition")
    c("COST", "AMZN", 0.85, "Costco membership vs Amazon Prime subscription retail")
    c("DLTR", "ROST", 0.85, "Discount dollar store vs off-price apparel retail")
    c("DLTR", "WMT", 0.85, "Budget retail and essential household grocery competition")
    c("ROST", "TJX", 0.95, "Direct rivalry in off-price branded apparel and home fashion")
    c("TJX", "TGT", 0.8, "Apparel and home decor consumer spending rivalry")
    c("HD", "LOW", 0.98, "Duopoly battle across home improvement, lumber, and hardware")
    s("SHW", "HD", 0.95, "Exclusive architectural paint distribution and retail supply")
    s("SHW", "LOW", 0.95, "Valspar and Sherwin-Williams paint products supply")
    c("ORLY", "CPRT", 0.8, "Automotive aftermarket maintenance and salvage parts ecosystem")
    c("ORLY", "HD", 0.75, "Automotive aftermarket maintenance parts vs hardware retail")

    # Automobiles & EV
    c("TSLA", "GM", 0.9, "Electric vehicle leadership vs Detroit legacy automakers")
    c("TSLA", "F", 0.9, "EV trucks (Cybertruck vs F-150 Lightning) and autonomous software")
    c("GM", "F", 0.95, "Historic American automotive rivalry in pickup trucks and SUVs")
    s("TXN", "TSLA", 0.85, "Analog chips and battery management silicon for Tesla EVs")
    s("ADI", "GM", 0.85, "Wireless battery management systems (wBMS) for Ultium EVs")
    c("UBER", "TSLA", 0.85, "Human rideshare network vs upcoming Robotaxi autonomous fleet")

    # Fast Food, Coffee & Athletic
    c("MCD", "SBUX", 0.85, "Breakfast food, drive-thru beverages, and retail restaurant footprint")
    c("NKE", "LULU", 0.9, "Athletic apparel, footwear, and athleisure lifestyle rivalry")
    s("NKE", "AMZN", 0.85, "Official digital brand presence and retail distribution")

    # Travel, Lodging & Mobility
    c("BKNG", "EXPE", 0.95, "Global online travel agency duopoly in hotel and flight reservations")
    c("BKNG", "ABNB", 0.9, "Hotel bookings vs alternative short-term vacation rentals")
    c("EXPE", "ABNB", 0.9, "Vrbo vacation home rentals vs Airbnb platform")
    c("ABNB", "MAR", 0.85, "Home-sharing rentals vs luxury and business hotel chains")
    c("MAR", "BKNG", 0.85, "Direct hotel loyalty booking vs third-party OTA commissions")
    c("DASH", "UBER", 0.95, "Fierce rivalry in food delivery (DoorDash vs Uber Eats)")
    s("DASH", "MCD", 0.85, "Primary restaurant delivery partner for McDonald's")
    s("DASH", "SBUX", 0.85, "On-demand delivery integration for Starbucks coffee")
    infra("UBER", "GOOGL", 0.9, "Google Cloud and Google Maps API navigation infrastructure")
    c("MELI", "AMZN", 0.85, "Latin America e-commerce and fintech dominance vs Amazon")
    c("PDD", "AMZN", 0.85, "Cross-border discount shopping (Temu vs Amazon Marketplace)")
    c("PDD", "MELI", 0.8, "Cross-border e-commerce in Latin America and emerging markets")

    # Real estate retail malls
    c("SPG", "MAR", 0.8, "Commercial retail shopping centers and luxury hotel destinations")
    c("SPG", "PLD", 0.75, "Commercial real estate REIT sector peers")
    s("SPG", "TGT", 0.85, "Shopping center anchor tenant leasing")
    s("SPG", "ROST", 0.85, "Retail outlet center store locations")

    # Consumer Staples (Beverages & Snacks)
    c("KO", "PEP", 0.98, "Historic cola wars and global non-alcoholic beverage rivalry")
    c("PEP", "MDLZ", 0.85, "Global salty snacks (Frito-Lay) vs sweet snacks (Oreo/Cadbury)")
    c("KO", "MNST", 0.9, "Coca-Cola is primary global distributor and minority owner of Monster")
    c("MNST", "PEP", 0.85, "Energy drinks competition (Monster vs Rockstar/Celsius)")
    c("KDP", "KO", 0.85, "Carbonated soft drinks and commercial beverage distribution")
    c("KDP", "PEP", 0.85, "Bottled soft drinks and tea distribution competition")
    c("CCEP", "KO", 0.95, "Largest independent European bottling partner for The Coca-Cola Co")
    c("CCEP", "PEP", 0.8, "European beverage bottling and distribution competition")
    c("MDLZ", "KHC", 0.85, "Packaged consumer foods and grocery retail shelf space")
    s("KHC", "WMT", 0.95, "Kraft Heinz packaged grocery supply to Walmart supercenters")
    s("KHC", "COST", 0.9, "Bulk condiments and packaged food supplied to Costco")
    c("PG", "CL", 0.9, "Personal care, oral hygiene, and household cleaning products rivalry")
    s("PG", "COST", 0.95, "Kirkland Signature co-packing and branded goods to Costco")
    s("CL", "WMT", 0.9, "Colgate oral care and Palmolive dish soap to Walmart")
    c("PM", "MO", 0.95, "Global smoke-free nicotine (IQOS/ZYN) vs domestic US tobacco")
    s("MO", "WMT", 0.85, "Nicotine products retail distribution in convenience locations")

    # Streaming & Media
    c("NFLX", "DIS", 0.95, "Subscription streaming video dominance (Netflix vs Disney+)")
    c("NFLX", "WBD", 0.85, "Streaming entertainment and original content rivalry (Netflix vs Max)")
    c("DIS", "WBD", 0.9, "Box office movies, television studios, and sports broadcasting")
    c("DIS", "CMCSA", 0.9, "Theme parks (Disney World vs Universal) and film studios")
    c("WBD", "CMCSA", 0.85, "Hollywood movie studios and television broadcasting competition")
    c("CMCSA", "CHTR", 0.9, "Major broadband, cable television, and MVNO mobile rivalry")
    c("TMUS", "CMCSA", 0.85, "5G wireless carriers vs cable broadband mobile MVNOs")
    c("EA", "DIS", 0.8, "Interactive gaming licensing (Star Wars, Marvel titles)")
    c("EA", "MSFT", 0.85, "EA Play subscription partnership and Xbox game catalog")
    c("EA", "WBD", 0.8, "Entertainment gaming and interactive software rivalry")

    # =========================================================================
    # 10. BROAD BENCHMARK INDICES & SECTOR ETFS
    # =========================================================================
    # S&P 500 ETF (SPY) Top Holdings
    p("SPY", "MSFT", 0.95, "Top tech component in S&P 500 index")
    p("SPY", "AAPL", 0.95, "Top consumer tech component in S&P 500 index")
    p("SPY", "NVDA", 0.95, "Top semiconductor and AI component in S&P 500 index")
    p("SPY", "AMZN", 0.9, "Top consumer discretionary component in S&P 500 index")
    p("SPY", "META", 0.9, "Top communication services component in S&P 500 index")
    p("SPY", "GOOGL", 0.9, "Top search and cloud component in S&P 500 index")
    p("SPY", "BRK.B", 0.85, "Top financial conglomerate component in S&P 500 index")
    p("SPY", "JPM", 0.85, "Top banking component in S&P 500 index")
    p("SPY", "LLY", 0.85, "Top healthcare pharmaceutical component in S&P 500 index")
    p("SPY", "XOM", 0.85, "Top energy component in S&P 500 index")

    # Nasdaq-100 ETF (QQQ) Top Holdings
    p("QQQ", "NVDA", 0.95, "Leading momentum driver in Nasdaq 100")
    p("QQQ", "AAPL", 0.95, "Top equity weighting in Nasdaq 100")
    p("QQQ", "MSFT", 0.95, "Top enterprise cloud weighting in Nasdaq 100")
    p("QQQ", "AMZN", 0.9, "Major tech and e-commerce driver in Nasdaq 100")
    p("QQQ", "META", 0.9, "Major communication driver in Nasdaq 100")
    p("QQQ", "AVGO", 0.9, "Major semiconductor driver in Nasdaq 100")
    p("QQQ", "TSLA", 0.85, "Major growth and EV driver in Nasdaq 100")
    p("QQQ", "COST", 0.85, "Top consumer staples holding in Nasdaq 100")
    p("QQQ", "NFLX", 0.85, "Top streaming media holding in Nasdaq 100")
    p("QQQ", "AMD", 0.85, "Top semiconductor growth holding in Nasdaq 100")

    # Dow Jones Industrial Average ETF (DIA) Top Holdings
    p("DIA", "UNH", 0.95, "Highest price-weighted stock in Dow Jones Industrial Average")
    p("DIA", "GS", 0.9, "Major investment banking component in Dow Jones Industrial Average")
    p("DIA", "MSFT", 0.9, "Leading technology component in Dow Jones Industrial Average")
    p("DIA", "HD", 0.85, "Retail component in Dow Jones Industrial Average")
    p("DIA", "CAT", 0.85, "Machinery component in Dow Jones Industrial Average")
    p("DIA", "MCD", 0.85, "Consumer franchise component in Dow Jones Industrial Average")
    p("DIA", "BA", 0.85, "Aerospace component in Dow Jones Industrial Average")
    p("DIA", "V", 0.85, "Payments component in Dow Jones Industrial Average")
    p("DIA", "AMGN", 0.85, "Biotechnology component in Dow Jones Industrial Average")

    # Small-Cap Market Breadth Benchmark (IWM)
    p("IWM", "SPY", 0.9, "US small-cap vs large-cap equity market breadth comparison")
    p("IWM", "QQQ", 0.85, "Small-cap equity vs large-cap tech relative performance")

    # Sector ETFs to sector champions
    p("SMH", "NVDA", 0.95, "Top weighting in VanEck Semiconductor ETF")
    p("SMH", "TSM", 0.95, "Foundry anchor in VanEck Semiconductor ETF")
    p("SMH", "AVGO", 0.9, "Custom silicon anchor in VanEck Semiconductor ETF")
    p("SMH", "ASML", 0.9, "Lithography anchor in VanEck Semiconductor ETF")
    p("SMH", "AMD", 0.9, "Compute anchor in VanEck Semiconductor ETF")
    p("SMH", "QCOM", 0.85, "Wireless anchor in VanEck Semiconductor ETF")

    p("XLF", "JPM", 0.95, "Top holding in Financial Select Sector SPDR")
    p("XLF", "BRK.B", 0.95, "Major conglomerate holding in Financial Sector SPDR")
    p("XLF", "BAC", 0.9, "Major bank holding in Financial Sector SPDR")
    p("XLF", "V", 0.9, "Top payments holding in Financial Sector SPDR")
    p("XLF", "MA", 0.9, "Major payment rails holding in Financial Sector SPDR")

    p("XLE", "XOM", 0.98, "Dominant upstream holding in Energy Select Sector SPDR")
    p("XLE", "CVX", 0.95, "Major integrated holding in Energy Select Sector SPDR")
    p("XLE", "COP", 0.9, "Top E&P holding in Energy Select Sector SPDR")
    p("XLE", "SLB", 0.85, "Top oilfield services holding in Energy Select Sector SPDR")

    p("XLV", "LLY", 0.95, "Top holding in Health Care Select Sector SPDR")
    p("XLV", "UNH", 0.95, "Top managed care holding in Health Care Sector SPDR")
    p("XLV", "JNJ", 0.9, "Top diversified healthcare holding in Health Care Sector SPDR")
    p("XLV", "ABBV", 0.9, "Major biopharma holding in Health Care Sector SPDR")

    p("XLI", "GE", 0.95, "Top aerospace holding in Industrial Select Sector SPDR")
    p("XLI", "CAT", 0.95, "Top machinery holding in Industrial Select Sector SPDR")
    p("XLI", "RTX", 0.9, "Top defense holding in Industrial Select Sector SPDR")
    p("XLI", "UNP", 0.9, "Top railroad holding in Industrial Select Sector SPDR")

    p("XLU", "NEE", 0.95, "Top holding in Utilities Select Sector SPDR")
    p("XLU", "SO", 0.9, "Major regulated utility holding in Utilities Sector SPDR")
    p("XLU", "CEG", 0.9, "Top nuclear clean energy holding in Utilities Sector SPDR")
    p("XLU", "VST", 0.9, "Top merchant power producer in Utilities Sector SPDR")

    p("XLK", "MSFT", 0.95, "Top holding in Technology Select Sector SPDR")
    p("XLK", "AAPL", 0.95, "Top consumer tech holding in Technology Sector SPDR")
    p("XLK", "NVDA", 0.95, "Top semiconductor holding in Technology Sector SPDR")

    p("XLP", "PG", 0.95, "Top holding in Consumer Staples Select Sector SPDR")
    p("XLP", "COST", 0.95, "Top warehouse retailer in Consumer Staples Sector SPDR")
    p("XLP", "WMT", 0.95, "Top retail hypermarket in Consumer Staples Sector SPDR")
    p("XLP", "KO", 0.9, "Top beverage holding in Consumer Staples Sector SPDR")

    p("XLY", "AMZN", 0.95, "Top holding in Consumer Discretionary Select Sector SPDR")
    p("XLY", "TSLA", 0.95, "Top EV and clean energy holding in Discretionary Sector SPDR")
    p("XLY", "HD", 0.9, "Top home improvement holding in Discretionary Sector SPDR")

    return edges


def populate_market_universe(graph: StockGraph) -> None:
    """Populate a StockGraph instance with the full NASDAQ-100 and S&P 500 universe."""
    for node in get_market_nodes():
        graph.add_node(node)
    for edge in get_market_edges():
        graph.add_edge(edge)
