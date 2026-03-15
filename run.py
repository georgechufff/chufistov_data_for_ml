from agents.data_collection_agent import DataCollectionAgent


if __name__ == "__main__":

    agent = DataCollectionAgent()

    df = agent.run(
        sources=[
            {
                'type': 'kaggle_dataset',
                'dataset_name': 'athugodage/russian-legal-text-parallel-corpus',
                'file_path': 'legal_text_corpus.csv'
            },
            {
                'type': 'scrape',
                'url': 'https://base.garant.ru/12150845/',
                'name': 'ЛК РФ'
            },
        ]
    )
    
    df.to_csv('corpus.csv', index=False)