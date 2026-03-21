from agents import DataCollectionAgent, DataQualityAgent

from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":

    collection_agent = DataCollectionAgent()
    quality_agent = DataQualityAgent(use_llm=True)

    # df = agent.run(
    #     sources=[
    #         {
    #             'type': 'kaggle_dataset',
    #             'dataset_name': 'athugodage/russian-legal-text-parallel-corpus',
    #             'file_path': 'legal_text_corpus.csv'
    #         },
    #         {
    #             'type': 'scrape',
    #             'url': 'https://base.garant.ru/12150845/',
    #             'name': 'ЛК РФ'
    #         },
    #     ]
    # )
    
    # df.to_csv('corpus.csv', index=False)
    
    quality_agent.run()
    