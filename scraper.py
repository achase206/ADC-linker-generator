# https://adcdb.idrblab.net/search/result/linker?search_api_fulltext=

from bs4 import BeautifulSoup
import requests
import time
import numpy as np
import pandas as pd


def scrape_adc(smiles_string):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    search_base = "https://adcdb.idrblab.net/search/result/linker?search_api_fulltext="

    try:
        url = search_base + smiles_string
        response = requests.get(url, headers=headers)
        html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")
        search_results = soup.find_all("div", class_="search-result-unit")

        if not search_results:
            return {
                "smiles": smiles_string,
                "linker_name": None,
                "adc_name": None,
                "drug_status": None,
                "indication": None,
                "antibody_name": None,
                "payload_name": None,
            }

        else:
            for result in search_results:
                data = {"smiles": smiles_string}

                b_tags = result.find_all("b")

                for b in b_tags:
                    key = b.get_text(strip=True)
                    value_node = b.next_sibling

                    # Check if the next sibling is a NavigableString (i.e., text)
                    if value_node and isinstance(value_node, str):
                        value = value_node.strip()

                        if key == "Linker Name:":
                            data["linker_name"] = value
                        elif key == "Antibody Drug Conjugate:":
                            data["adc_name"] = value
                        elif key == "Drug Status:":
                            data["drug_status"] = value
                        elif key == "Representative Indication:":
                            data["indication"] = value
                        elif key == "Antibody Name:":
                            data["antibody_name"] = value
                        elif key == "Payload Name:":
                            data["payload_name"] = value
            return data

    except requests.exceptions.HTTPError as http_err:
        print(f"  > HTTP error for {smiles_string}: {http_err}")
        return {
            "smiles": smiles_string,
            "linker_name": None,
            "adc_name": None,
            "drug_status": None,
            "indication": None,
            "antibody_name": None,
            "payload_name": None,
        }
    except requests.exceptions.RequestException as err:
        print(f"  > Error fetching {smiles_string}: {err}")
        return {
            "smiles": smiles_string,
            "linker_name": None,
            "adc_name": None,
            "drug_status": None,
            "indication": None,
            "antibody_name": None,
            "payload_name": None,
        }
    except Exception as e:
        print(f"  > An unknown error occurred for {smiles_string}: {e}")
        return {
            "smiles": smiles_string,
            "linker_name": None,
            "adc_name": None,
            "drug_status": None,
            "indication": None,
            "antibody_name": None,
            "payload_name": None,
        }


def run_scrape(smiles_list):
    scrape_results = []
    counter = 1
    for smiles in smiles_list:
        print(f"{counter}/{len(smiles_list)} processing...")
        scrape_results.append(scrape_adc(smiles))
        sleep_time = np.random.uniform(1, 2)
        # print(f"waiting {sleep_time} seconds..")
        time.sleep(sleep_time)
        counter += 1
    return scrape_results


if __name__ == "__main__":

    adc_df = pd.read_csv("data/linkers.csv", encoding="latin1")
    smiles_list = adc_df["smiles"].to_list()
    batch_size = 100

    # smiles_list = [
    #     "CC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H](CCCCNC(=O)OCC=C)C(=O)NC2=CC=C(C=C2)COC(=O)OC3=CC=C(C=C3)[N+](=O)[O-]",
    #     "CC(C)[C@@H](C(=O)N[C@@H](CCCNC(=O)N)C(=O)NC1=CC=C(C=C1)CO)NC(=O)OCC2C3=CC=CC=C3C4=CC=CC=C24",
    # ]

    for i in range(0, len(smiles_list), batch_size):
        batch_result = run_scrape(smiles_list[i : i + batch_size])
        batch_df = pd.DataFrame(batch_result)
        batch_df.to_pickle(f"batches/adc_batch_{i}.pkl")
