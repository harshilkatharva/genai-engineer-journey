from uuid import UUID

import psycopg
from ai_app.models.usage_report import UsageBreakdown, UsageReport


class DBReport:
    async def get_user_usage_report(
        self,
        user_id: UUID,
    ) -> UsageReport:
        query = """
            SELECT
                feature,
                COUNT(*) AS message_count,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(estimated_cost), 0) AS estimated_cost
            FROM history
            WHERE user_id = %s
            GROUP BY feature
            ORDER BY feature
        """

        async with (
            await psycopg.AsyncConnection.connect(self.connection_string) as conn,
            conn.cursor() as cursor,
        ):
            await cursor.execute(query, (user_id,))
            rows = await cursor.fetchall()

        breakdown = [
            UsageBreakdown(
                name=feature,
                message_count=message_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=float(estimated_cost),
            )
            for (
                feature,
                message_count,
                input_tokens,
                output_tokens,
                estimated_cost,
            ) in rows
        ]

        return UsageReport(
            total_messages=sum(item.message_count for item in breakdown),
            total_input_tokens=sum(item.input_tokens for item in breakdown),
            total_output_tokens=sum(item.output_tokens for item in breakdown),
            total_estimated_cost=sum(item.estimated_cost for item in breakdown),
            breakdown=breakdown,
        )

    async def get_feature_usage_report(
        self,
        feature: str,
    ) -> UsageReport:
        query = """
            SELECT
                user_id,
                COUNT(*) AS message_count,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(estimated_cost), 0) AS estimated_cost
            FROM history
            WHERE feature = %s
            GROUP BY user_id
            ORDER BY user_id
        """

        async with (
            await psycopg.AsyncConnection.connect(self.connection_string) as conn,
            conn.cursor() as cursor,
        ):
            await cursor.execute(query, (feature,))
            rows = await cursor.fetchall()

        breakdown = [
            UsageBreakdown(
                name=user_id,
                message_count=message_count,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=float(estimated_cost),
            )
            for (
                user_id,
                message_count,
                input_tokens,
                output_tokens,
                estimated_cost,
            ) in rows
        ]

        return UsageReport(
            total_messages=sum(item.message_count for item in breakdown),
            total_input_tokens=sum(item.input_tokens for item in breakdown),
            total_output_tokens=sum(item.output_tokens for item in breakdown),
            total_estimated_cost=sum(item.estimated_cost for item in breakdown),
            breakdown=breakdown,
        )
