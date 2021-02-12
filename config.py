from collections import namedtuple

# create a named tuple which we can use to create locations of the
# input document which we wish to OCR
OCRLocation = namedtuple("OCRLocation", ["id", "bbox",
    "filter_keywords"])

# define the locations of each area of the document we wish to OCR

OCR_LOCATIONS = [
    OCRLocation("Exporter", (252, 147, 1368, 349),
        ['Exporter', '(Name,', 'full', 'address', 'country)']),
    OCRLocation("preferential_trade_between", (1720, 416, 1435, 511),
        ['Certificate', 'used', 'in', 'preferential', 'trade', 'between', "and", "(Insert", 		'appropriate', 'countries', 'or', 'groups', 'of', 'territories)']),
    OCRLocation("Consignee", (252, 546, 1368, 615),
        ["Consignee", "(Name", "full address", "country)", ",", "(Optional)", "3"]),
    OCRLocation("transport_details", (252, 1211, 1368, 463),
        ["Transport", "details", "(Optional)", "6"]),
    OCRLocation("item_number", (252, 1724, 772, 1799),
        ["8", "Item", "number:", "marks", "and", "numbers"]),
    OCRLocation("description_of_goods", (1024, 1724, 1422, 1799),
        ["Number", "and", "kind", "of", "packages", "(1):", "description", "goods"]),
    OCRLocation("gross_weight", (2466, 1724, 349, 1799), 
        ["Gross", "weight", "(kg)", "or", "other", "measure", "(litres,", "cu.", "m.,", "etc)"]),
    OCRLocation("customs_office", (252, 3743, 1797, 792),
        ["Customs", "11.", "office", 'Declaration', 'certified', 'Export', 'document', '(2):', 'From', 'No.', 'Endorsement', 'stamp', 'Issuing', 'country', 'or', 'territory', 'UNITED', 'KINGDOM', "date", "(Signature)"]),
    OCRLocation("exporter_date_signature", (2049, 3743, 1106, 792),
        ['12.', 'Declaration', 'by', 'Exporter', 'I,', 'the', 'undersigned,', 'declare', 'that', 'the', 'goods', 'described', 'above', 'meet', 'conditions', 'required', 'for', 'issue', 'of', 'this', 'certificate.',"(Place", "and", "date", "(Signature)", "."]),
    
]


ground_truth = {"Exporter":["Luke Skywalker", "Remote Island", "Ahch-To", "Outer Reaches"],
                "preferential_trade_between":["Ahch-To", "Jakko"],
                "Consignee":["Rey Palpatine", "Niima Outpost", "Jakko", "Western Reaches"],
                "transport_details":["Millennium Falcon"],
                "item_number":["#03418GH 093"],
                "description_of_goods":["Ancient Jedi texts"],
                "gross_weight":["15 kg"],
                "customs_office":["Ahch-To", "30327", "Niima Outpost Militia"],
                "customs_date_signature":["20/01/21", "Constable Zuvio"],
                "exporter_date_signature":["Jedi Temple, 01/12/20", "Luck Skywalker"]}

headings = ['1. Exporter \n(Name, full address, country)',
            '2. Certificate used in \npreferential trade between',
            '3. Consignee \n(Name, full address, country)(Optional)',
            '6. Transport details (Optional)',
            '8.1. Item number: \nmarks and numbers',
            '8.2. Number and kind of packages (1): \ndescription of goods',
            '9. Gross \nweight (kg) \nof other \nmeasure \n(liters, cu. m., etc)',
            '11.1. Customs Endorsement',
            '11.2. Customs Signature',
            '12. Declaration by the Exporter']
