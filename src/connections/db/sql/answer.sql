-- 移除CTE：将CTE改为子查询，减少临时结果集的内存占用
SELECT 
    t2.col_question_id, 
    t2.col_answer_id, 
    t2.col_answer_content, 
    t2.col_is_default_answer, 

    -- 使用JSON_ARRAYAGG收集标签列表
    (SELECT JSON_ARRAYAGG(label_type)
     FROM schema_name.table_answer_type_label 
     WHERE question_id = t2.col_question_id 
       AND answer_id = t2.col_answer_id 
       AND is_delete = 0
    ) AS answer_type_list,

    -- 简化标签的JSON结构
    (SELECT JSON_ARRAYAGG(
        JSON_OBJECT(
            'label_id', cl.label_id,
            'question_id', cl.question_id,
            'model_id', cl.model_id,
            'model_name', cl.model_name,
            'series_id', cl.series_id,
            'series_name', cl.series_name
        ))
     FROM schema_name.table_answer_item_label cl
     WHERE cl.question_id = t2.col_question_id 
       AND cl.answer_id = t2.col_answer_id
       AND cl.is_delete = 0
    ) AS item_label_list,

    t2.col_lowest_version, 
    t2.col_highest_version, 
    t2.col_valid_begin_time, 
    t2.col_valid_end_time,
    t2.col_status AS answer_status

FROM 
    schema_name.table_question_answer t2 
WHERE 
    t2.col_status = 1  
    AND t2.is_delete = 0
    AND (t2.col_valid_begin_time <= NOW() OR t2.col_valid_begin_time IS NULL)
    AND (t2.col_valid_end_time >= NOW() OR t2.col_valid_end_time IS NULL)

    -- 使用EXISTS替代JOIN，更高效
    AND EXISTS (
        SELECT 1 
        FROM schema_name.table_answer_type_label at_filter
        WHERE at_filter.question_id = t2.col_question_id 
          AND at_filter.answer_id = t2.col_answer_id
          AND at_filter.is_delete = 0
          AND at_filter.label_type IN (%s)  -- 直接比较比正则更高效
    );