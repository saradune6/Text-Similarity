-- CREATING THE MODEL
CREATE OR REPLACE MODEL `bharatpe-analytics-prod.bharatpe_ml_data.multimodalembedding`
  REMOTE WITH CONNECTION `projects/bharatpe-analytics-prod/locations/asia-south1/connections/image_embedding_conn`
  OPTIONS(ENDPOINT = 'multimodalembedding@001');


-- CREATING THE OBJECT TABLE
CREATE OR REPLACE EXTERNAL TABLE
  `bharatpe-analytics-prod.bharatpe_ml_data.external_images_table2`  -- Your project, dataset, and table name
  WITH CONNECTION `asia-south1.image_embedding_conn`  -- Your connection ID (replace with the actual one)
  OPTIONS(
    object_metadata = 'SIMPLE',  -- Use basic metadata
    uris = ['gs://bharatpe_ml_image_bucket_2/kyc_sept24_data3_entire/*']  -- URI path to your image files (adjust extension if needed)
  );



-- bharatpe_ml_image_bucket_2/kyc_sept24_data3_entire



-- EMBEDDING GENERATION
-- CREATE OR REPLACE TABLE `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings` AS
-- SELECT *
-- FROM ML.GENERATE_EMBEDDING(
--   MODEL `bharatpe-analytics-prod.bharatpe_ml_data.multimodalembedding`,  -- Replace with your model path
--   TABLE `bharatpe-analytics-prod.bharatpe_ml_data.external_images_table`,  -- Replace with your table containing the image data
--   STRUCT(
--     -- TRUE AS flatten_json,  -- Set to TRUE to parse the embedding into a separate column
--     1408 AS output_dimensionality  -- Set to 256 to get embeddings of size 256
--   )
-- );



-- EMBEDDDING GENERATION
CREATE OR REPLACE TABLE `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings2` AS
SELECT *
FROM ML.GENERATE_EMBEDDING(
  MODEL `bharatpe-analytics-prod.bharatpe_ml_data.multimodalembedding`,  -- Replace with your model path
  (  -- Wrap the query inside parentheses
    SELECT *
    FROM `bharatpe-analytics-prod.bharatpe_ml_data.external_images_table2`
    LIMIT 5000  -- Limit to the first 6000 images
  ),
  STRUCT(
    1408 AS output_dimensionality  -- Set to 1408 to get embeddings of size 1408
  )
);




-- SELECT * 
-- FROM `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings`
-- WHERE ARRAY_LENGTH(ml_generate_embedding_result) = 0;



-- DELETE FROM `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings`
-- WHERE ARRAY_LENGTH(ml_generate_embedding_result) = 0;

-- SELECT COUNT(*) AS total_rows
-- FROM `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings`;



-- INSERT INTO `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings`
-- SELECT * EXCEPT(row_num)  -- Exclude the extra column
-- FROM (
--   SELECT *, ROW_NUMBER() OVER () AS row_num
--   FROM `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings`
-- )
-- WHERE row_num = 1;





-- CREATE THE VECTOR INDEX ON THE REGULAR TABLE
CREATE OR REPLACE VECTOR INDEX image_index
ON `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings2` (ml_generate_embedding_result)
OPTIONS(distance_type='COSINE', index_type='IVF', ivf_options='{"num_lists": 10}');




select * from `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings2`






-- RUN A VECTOR SEARCH
-- SELECT
--    *
-- FROM
--     `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings`,
--     ML.GENERATE_EMBEDDING_RESULT AS embedding_col
-- FROM VECTOR SEARCH (
--     -- Subquery: Generate an embedding from text input
--     SELECT
--         ML.GENERATE_EMBEDDING(
--             MODEL `bharatpe-analytics-prod.bharatpe_ml_data.multimodalembedding`,
--             (SELECT "comfy rainbow sweaters" AS content),
--             STRUCT(TRUE AS flatten_json_output)
--         ) AS m1_generate_embedding_result
--     )
-- )
-- -- Specify the top-k results
-- TOP_K => 5;




SELECT
*
FROM VECTOR_SEARCH(
    TABLE `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings2`, 'ml_generate_embedding_result',  -- The table with the image embeddings
    (  -- Generate the embedding for the query image
      SELECT query.ml_generate_embedding_result AS query  -- Use the embedding column for the query image
      FROM ML.GENERATE_EMBEDDING(
        MODEL `bharatpe-analytics-prod.bharatpe_ml_data.multimodalembedding`,  -- The model for image embedding generation
        (SELECT uri, content_type  -- Selecting from the object table containing metadata
         FROM `bharatpe-analytics-prod.bharatpe_ml_data.external_images_table2`
         WHERE uri = 'gs://bharatpe_ml_image_bucket_2/kyc_sept24_data3_test/000009c9-6b78-4079-bcc3-89103165c8d7.jpg')  -- URI for the image to compare
      ) AS query
    ),
    top_k => 10  -- Get the top 5 most similar images
)





-- SELECT
--   COUNT(*) AS similar_images_count  -- Count the number of rows returned
-- FROM VECTOR_SEARCH(
--     TABLE `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings`, 'ml_generate_embedding_result',  -- The table with the image embeddings
--     (
--       SELECT query.ml_generate_embedding_result AS query  -- Use the embedding column for the query image
--       FROM ML.GENERATE_EMBEDDING(
--         MODEL `bharatpe-analytics-prod.bharatpe_ml_data.multimodalembedding`,  -- The model for image embedding generation
--         (SELECT uri, content_type  -- Selecting from the object table containing metadata
--          FROM `bharatpe-analytics-prod.bharatpe_ml_data.external_images_table`
--          WHERE uri = 'gs://bharatpe_ml_image_bucket_2/kyc_sept24_data3_test/000009c9-6b78-4079-bcc3-89103165c8d7.jpg')  -- URI for the image to compare
--       ) AS query
--     ),
--     top_k => 12  -- Get the top 5 most similar images
-- );







SELECT
  base.*,  -- Include all columns from the base table
  (1 - base.distance) * 100 AS similarity_percentage  -- Calculate similarity percentage
FROM VECTOR_SEARCH(
    TABLE `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings2`, 'ml_generate_embedding_result',  -- The table with the image embeddings
    (  -- Generate the embedding for the query image
      SELECT query.ml_generate_embedding_result AS query  -- Use the embedding column for the query image
      FROM ML.GENERATE_EMBEDDING(
        MODEL `bharatpe-analytics-prod.bharatpe_ml_data.multimodalembedding`,  -- The model for image embedding generation
        (SELECT uri, content_type  -- Selecting from the object table containing metadata
         FROM `bharatpe-analytics-prod.bharatpe_ml_data.external_images_table2`
         WHERE uri = 'gs://bharatpe_ml_image_bucket_2/kyc_sept24_data3_test/000009c9-6b78-4079-bcc3-89103165c8d7.jpg')  -- URI for the image to compare
      ) AS query
    ),
    top_k => 10  -- Get the top 10 most similar images
) AS base;  -- Alias for the VECTOR_SEARCH result











WITH embeddings AS (
  SELECT uri, ml_generate_embedding_result FROM `bharatpe-analytics-prod.bharatpe_ml_data.image_embeddings2`
),
comparison AS (
  SELECT
    query.uri AS query_uri,  -- URI of the query image
    base.uri AS compared_uri,  -- URI of the compared image
    SAFE_DIVIDE(
      (SELECT SUM(qv * bv)
       FROM UNNEST(query.ml_generate_embedding_result) AS qv WITH OFFSET pos1
       JOIN UNNEST(base.ml_generate_embedding_result) AS bv WITH OFFSET pos2
       ON pos1 = pos2),
      (SQRT((SELECT SUM(POW(qv, 2)) FROM UNNEST(query.ml_generate_embedding_result) AS qv)) *
       SQRT((SELECT SUM(POW(bv, 2)) FROM UNNEST(base.ml_generate_embedding_result) AS bv)))
    ) * 100 AS similarity_percentage  -- Cosine similarity as a percentage
  FROM embeddings AS query, embeddings AS base
  WHERE query.uri != base.uri  -- Exclude comparing the same image to itself
)
SELECT
  query_uri,
  ARRAY_AGG(STRUCT(compared_uri, similarity_percentage) ORDER BY similarity_percentage DESC LIMIT 10) AS top_10_similar
FROM comparison
GROUP BY query_uri;