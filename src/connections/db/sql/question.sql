SELECT 
    q.question_id, 
    q.question_content,
    q.question_type,    
    q.category_name, 
    q.valid_begin_time, 
    q.valid_end_time, 
    kb.base_name
FROM 
    schema_name.question_table q

-- 关联知识库表（可选，LEFT JOIN 可保留无匹配的记录）
LEFT JOIN schema_name.knowledge_base kb 
    ON q.base_id = kb.base_id 
    AND kb.is_delete = 0  

WHERE 
    q.status = 1  
    AND q.approval_status = 2  
    AND (q.valid_begin_time <= NOW() OR q.valid_begin_time IS NULL)
    AND (q.valid_end_time >= NOW() OR q.valid_end_time IS NULL)
    AND q.is_delete = 0  