/* eslint-disable */
// tslint:disable
/**
 * FastAPI
 *
 *
 * OpenAPI spec version: 0.1.0
 *
 *
 * NOTE: This class is auto generated by OpenAPI Generator+.
 * https://github.com/karlvr/openapi-generator-plus
 * Do not edit the class manually.
 */

export namespace Api {
	export interface Add201Response {
	}

	export interface AddEmbedding {
		embeddings?: Api.AddEmbedding.Embedding[];
		metadatas?: Api.AddEmbedding.Metadata[];
		documents?: string[];
		ids: string[];
		'increment_index'?: boolean;
	}

	/**
	 * @export
	 * @namespace AddEmbedding
	 */
	export namespace AddEmbedding {
		export interface Embedding {
		}

		export interface Metadata {
		}

	}

	export interface ADelete200Response {
	}

	export interface AGet200Response {
	}

	export interface Count200Response {
	}

	export interface CreateCollection {
		name: string;
		metadata?: Api.CreateCollection.Metadata;
		'get_or_create'?: boolean;
	}

	/**
	 * @export
	 * @namespace CreateCollection
	 */
	export namespace CreateCollection {
		export interface Metadata {
		}

	}

	export interface CreateCollection200Response {
	}

	export interface CreateIndex200Response {
	}

	export interface DeleteCollection200Response {
	}

	export interface DeleteEmbedding {
		ids?: string[];
		where?: Api.DeleteEmbedding.Where;
		'where_document'?: Api.DeleteEmbedding.WhereDocument;
	}

	/**
	 * @export
	 * @namespace DeleteEmbedding
	 */
	export namespace DeleteEmbedding {
		export interface Where {
		}

		export interface WhereDocument {
		}

	}

	export interface GetCollection200Response {
	}

	export interface GetEmbedding {
		ids?: string[];
		where?: Api.GetEmbedding.Where;
		'where_document'?: Api.GetEmbedding.WhereDocument;
		sort?: string;
		/**
		 * @type {number}
		 * @memberof GetEmbedding
		 */
		limit?: number;
		/**
		 * @type {number}
		 * @memberof GetEmbedding
		 */
		offset?: number;
		include?: (Api.GetEmbedding.Include.EnumValueEnum | Api.GetEmbedding.Include.EnumValueEnum2 | Api.GetEmbedding.Include.EnumValueEnum3 | Api.GetEmbedding.Include.EnumValueEnum4)[];
	}

	/**
	 * @export
	 * @namespace GetEmbedding
	 */
	export namespace GetEmbedding {
		export interface Where {
		}

		export interface WhereDocument {
		}

		export type Include = Api.GetEmbedding.Include.EnumValueEnum | Api.GetEmbedding.Include.EnumValueEnum2 | Api.GetEmbedding.Include.EnumValueEnum3 | Api.GetEmbedding.Include.EnumValueEnum4;

		/**
		 * @export
		 * @namespace Include
		 */
		export namespace Include {
			export enum EnumValueEnum {
				Documents = 'documents'
			}

			export enum EnumValueEnum2 {
				Embeddings = 'embeddings'
			}

			export enum EnumValueEnum3 {
				Metadatas = 'metadatas'
			}

			export enum EnumValueEnum4 {
				Distances = 'distances'
			}

		}

	}

	export interface GetNearestNeighbors200Response {
	}

	export interface Heartbeat200Response {
	}

	export interface HTTPValidationError {
		detail?: Api.ValidationError[];
	}

	export interface ListCollections200Response {
	}

	export interface QueryEmbedding {
		where?: Api.QueryEmbedding.Where;
		'where_document'?: Api.QueryEmbedding.WhereDocument;
		'query_embeddings': Api.QueryEmbedding.QueryEmbedding2[];
		/**
		 * @type {number}
		 * @memberof QueryEmbedding
		 */
		'n_results'?: number;
		include?: (Api.QueryEmbedding.Include.EnumValueEnum | Api.QueryEmbedding.Include.EnumValueEnum2 | Api.QueryEmbedding.Include.EnumValueEnum3 | Api.QueryEmbedding.Include.EnumValueEnum4)[];
	}

	/**
	 * @export
	 * @namespace QueryEmbedding
	 */
	export namespace QueryEmbedding {
		export interface Where {
		}

		export interface WhereDocument {
		}

		export interface QueryEmbedding2 {
		}

		export type Include = Api.QueryEmbedding.Include.EnumValueEnum | Api.QueryEmbedding.Include.EnumValueEnum2 | Api.QueryEmbedding.Include.EnumValueEnum3 | Api.QueryEmbedding.Include.EnumValueEnum4;

		/**
		 * @export
		 * @namespace Include
		 */
		export namespace Include {
			export enum EnumValueEnum {
				Documents = 'documents'
			}

			export enum EnumValueEnum2 {
				Embeddings = 'embeddings'
			}

			export enum EnumValueEnum3 {
				Metadatas = 'metadatas'
			}

			export enum EnumValueEnum4 {
				Distances = 'distances'
			}

		}

	}

	export interface RawSql {
		'raw_sql': string;
	}

	export interface RawSql200Response {
	}

	export interface Reset200Response {
	}

	export interface Root200Response {
	}

	export interface Update200Response {
	}

	export interface UpdateCollection {
		'new_name'?: string;
		'new_metadata'?: Api.UpdateCollection.NewMetadata;
	}

	/**
	 * @export
	 * @namespace UpdateCollection
	 */
	export namespace UpdateCollection {
		export interface NewMetadata {
		}

	}

	export interface UpdateCollection200Response {
	}

	export interface UpdateEmbedding {
		embeddings?: Api.UpdateEmbedding.Embedding[];
		metadatas?: Api.UpdateEmbedding.Metadata[];
		documents?: string[];
		ids: string[];
		'increment_index'?: boolean;
	}

	/**
	 * @export
	 * @namespace UpdateEmbedding
	 */
	export namespace UpdateEmbedding {
		export interface Embedding {
		}

		export interface Metadata {
		}

	}

	export interface Upsert200Response {
	}

	export interface ValidationError {
		loc: (string | number)[];
		msg: string;
		'type': string;
	}

	export interface Version200Response {
	}

}
