## Configuration Structure and Parameters

The configuration is organized under a single top-level section for client-related settings.

### `client`
* **Description:** Settings for connecting to the data service or client API.
    * `datastack_name`
        * **Type:** `string`
        * **Description:** The name of the datastack to connect to within the CAVE (Connectome Annotation Versioning Engine) framework. This specifies the particular dataset version or instance to be used.
        * **Example:** `"minnie65_phase3_v1"`
    * `my_token`
        * **Type:** `string`
        * **Description:** Your personal authentication token for accessing the datastack. This should typically be kept secure and might be loaded from an environment variable or a separate, non-versioned file.
        * **Note:** In the provided example, it's an empty string, indicating it needs to be filled in.
    * `mat_version_start`
        * **Type:** `integer`
        * **Description:** The starting materialization version.
    * `mat_version_end`
        * **Type:** `integer`
        * **Description:** The ending materialization version. This defines the latest version of the data that the client should consider.