import json


def normalize_value(value):
    """
    Normalisiert einen Wert für den Vergleich.
    Konvertiert Zahlen-Strings in Zahlen und berücksichtigt verschiedene Darstellungen.

    Args:
        value: Der zu normalisierende Wert

    Returns:
        Normalisierter Wert
    """
    # Wenn der Wert ein String ist, versuche ihn in eine Zahl zu konvertieren
    if isinstance(value, str):
        # Entferne Leerzeichen
        value = value.strip()

        # Versuche Konvertierung in Float
        try:
            return float(value)
        except ValueError:
            # Wenn Konvertierung fehlschlägt, behalte den String bei
            return value

    return value


def compare_json_arrays(json1, json2, fields_to_compare=None, exact_match=False, debug=True):
    """
    Vergleicht zwei JSON-Arrays und prüft, welche Einträge in den spezifizierten Feldern übereinstimmen.

    Args:
        json1 (list): Erstes JSON-Array
        json2 (list): Zweites JSON-Array
        fields_to_compare (list): Felder, die verglichen werden sollen (optional)
        exact_match (bool): Ob ein exakter Vergleich durchgeführt werden soll
        debug (bool): Ob Debug-Informationen ausgegeben werden sollen

    Returns:
        dict: Enthält Zähler für übereinstimmende und nicht übereinstimmende Einträge sowie Details
    """
    # Standardfelder, wenn keine angegeben sind
    if fields_to_compare is None:
        fields_to_compare = ["temperature", "r1", "r2", "monomer1_s", "monomer2_s"]  # Nur Temperatur vergleichen

    # Counter für Matches und Non-Matches
    match_counter = 0
    non_match_counter = 0

    # Liste für nicht übereinstimmende Einträge
    non_matching_entries = []
    matching_entries = []

    # Debug: Einträge in beiden Arrays ausgeben
    if debug:
        print(f"Anzahl Einträge in json1: {len(json1)}")
        print(f"Anzahl Einträge in json2: {len(json2)}")
        print(f"Zu vergleichende Felder: {fields_to_compare}")

        # Stichprobe der Werte für die zu vergleichenden Felder
        print("\nStichprobe der Werte in json1:")
        for field in fields_to_compare:
            values = [entry.get(field) for entry in json1[:5] if field in entry]
            print(f"  {field}: {values}")

        print("\nStichprobe der Werte in json2:")
        for field in fields_to_compare:
            values = [entry.get(field) for entry in json2[:5] if field in entry]
            print(f"  {field}: {values}")

    # Durchlaufe alle Einträge im ersten JSON-Array
    for i, entry1 in enumerate(json1):
        # Prüfe für jeden Eintrag im zweiten JSON-Array
        entry_matched = False

        for j, entry2 in enumerate(json2):
            # Prüfe, ob alle relevanten Felder übereinstimmen
            all_fields_match = True
            mismatched_fields = []

            for field in fields_to_compare:
                # Überprüfe, ob das Feld in beiden Einträgen vorhanden ist
                if field not in entry1 or field not in entry2:
                    all_fields_match = False
                    mismatched_fields.append({
                        "field": field,
                        "value1": entry1.get(field, "Nicht vorhanden"),
                        "value2": entry2.get(field, "Nicht vorhanden")
                    })
                    continue

                # Normalisiere die Werte für den Vergleich
                value1 = normalize_value(entry1[field])
                value2 = normalize_value(entry2[field])

                # Wenn eines der Felder nicht übereinstimmt, setze all_fields_match auf False
                if exact_match:
                    # Exakter Vergleich
                    if value1 != value2:
                        all_fields_match = False
                        mismatched_fields.append({
                            "field": field,
                            "value1": entry1[field],
                            "value2": entry2[field],
                            "normalized1": value1,
                            "normalized2": value2
                        })
                else:
                    # Toleranter Vergleich für numerische Werte
                    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                        # Toleranz für Float-Vergleiche
                        if abs(value1 - value2) > 0.0001:
                            all_fields_match = False
                            mismatched_fields.append({
                                "field": field,
                                "value1": entry1[field],
                                "value2": entry2[field],
                                "normalized1": value1,
                                "normalized2": value2,
                                "difference": abs(value1 - value2)
                            })
                    elif value1 != value2:
                        all_fields_match = False
                        mismatched_fields.append({
                            "field": field,
                            "value1": entry1[field],
                            "value2": entry2[field],
                            "normalized1": value1,
                            "normalized2": value2
                        })

            # Wenn alle Felder übereinstimmen, erhöhe den Match-Counter
            if all_fields_match:
                entry_matched = True
                matching_entries.append({
                    "entry1_index": i,
                    "entry2_index": j,
                    "entry1_id": entry1.get("_id", "Keine ID"),
                    "entry2_id": entry2.get("_id", "Keine ID"),
                    "matched_fields": {field: entry1.get(field) for field in fields_to_compare}
                })
                match_counter += 1
                if debug:
                    print(f"Match gefunden: Entry1[{i}] mit Entry2[{j}]")
                    for field in fields_to_compare:
                        print(f"  {field}: {entry1.get(field)} = {entry2.get(field)}")
                break  # Keine weiteren Einträge im zweiten JSON prüfen

        # Wenn kein Match gefunden wurde, erhöhe den Non-Match-Counter
        if not entry_matched:
            non_match_counter += 1

            # Speichere den nicht übereinstimmenden Eintrag
            non_matching_entries.append({
                "entry_index": i,
                "id": entry1.get("_id", "Keine ID"),
                "filename": entry1.get("filename", "Kein Dateiname"),
                "field_values": {field: entry1.get(field) for field in fields_to_compare}
            })

    return {
        "match_counter": match_counter,
        "non_match_counter": non_match_counter,
        "matching_entries": matching_entries,
        "non_matching_entries": non_matching_entries
    }


def main():
    """
    Hauptfunktion zum Vergleichen zweier JSON-Dateien.
    """
    # Dateipfade zu den JSON-Dateien
    file1 = "data_new.json"
    file2 = "data_old.json"

    try:
        # JSON-Daten laden
        json1 = load_json_from_file(file1)
        json2 = load_json_from_file(file2)

        # Überprüfen, ob es sich um Arrays handelt
        if not isinstance(json1, list):
            json1 = [json1]
        if not isinstance(json2, list):
            json2 = [json2]

        # Felder zum Vergleichen definieren - hier nur Temperatur
        fields_to_compare = ["monomer1_s", "monomer2_s"]

        # JSON-Arrays vergleichen mit Debug-Informationen
        result = compare_json_arrays(json1, json2, fields_to_compare, exact_match=False, debug=True)

        # Ergebnisse ausgeben
        print(f"\nÜbereinstimmende Einträge: {result['match_counter']}")
        print(f"Nicht übereinstimmende Einträge: {result['non_match_counter']}")

        # Details zu übereinstimmenden Einträgen ausgeben
        if result['match_counter'] > 0:
            print("\nDetails zu übereinstimmenden Einträgen:")
            for i, entry in enumerate(result['matching_entries'][:10], 1):  # Zeige nur die ersten 10
                print(f"\n{i}. ID1: {entry['entry1_id']} - ID2: {entry['entry2_id']}")
                for field, value in entry['matched_fields'].items():
                    print(f"   {field}: {value}")

            if len(result['matching_entries']) > 10:
                print(f"... und {len(result['matching_entries']) - 10} weitere Übereinstimmungen")

        # Details zu nicht übereinstimmenden Einträgen ausgeben
        if result['non_match_counter'] > 0:
            print("\nDetails zu nicht übereinstimmenden Einträgen (Stichprobe):")
            sample_size = min(10, len(result['non_matching_entries']))
            for i, entry in enumerate(result['non_matching_entries'][:sample_size], 1):
                print(f"\n{i}. ID: {entry['id']}")
                for field, value in entry['field_values'].items():
                    print(f"   {field}: {value}")

            if len(result['non_matching_entries']) > sample_size:
                print(f"... und {len(result['non_matching_entries']) - sample_size} weitere ohne Übereinstimmung")

        print("\nVergleich abgeschlossen.")

    except Exception as e:
        print(f"Fehler beim Vergleichen der JSON-Dateien: {e}")
        import traceback
        traceback.print_exc()


def load_json_from_file(file_path):
    """
    Lädt JSON-Daten aus einer Datei.

    Args:
        file_path (str): Pfad zur JSON-Datei

    Returns:
        list or dict: Die geladenen JSON-Daten
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    main()